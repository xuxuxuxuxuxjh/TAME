#!/usr/bin/env python
# coding=utf-8
"""
Preprocess sepsis time-series data into the file format expected by TAME.

This script supports two modes:
- Legacy MIMIC-III mode: consumes raw MIMIC-III tables + pivoted_*.csv.
- Extracted sepsis mode: consumes already-extracted hourly time series data
  (e.g. `sepsis_timeseries_hourly.csv`) and directly writes:
    - <data-dir>/<dataset>/train_groundtruth/<stay_id>.csv
    - <data-dir>/<dataset>/train_with_missing/<stay_id>.csv

For extracted sepsis mode, point `--mimic-dir` to the folder containing
`sepsis_timeseries_hourly.csv` (e.g. mimic-code export folder).
"""

from __future__ import print_function

import csv
import os
import sys
import time
import json
import random
import errno
from glob import glob
from collections import OrderedDict

import numpy as np
from tqdm import tqdm

sys.path.append('../tools')
import parse, py_op

args = parse.args


variable_map_dict = {
            # lab
            'WBC': 'wbc',
            'bun': 'bun',
            'sodium': 'sodium',
            'pt': 'pt',
            'INR': 'inr',
            'PTT': 'ptt',
            'platelet': 'platelet',
            'lactate' : 'lactate',
            'hemoglobin': 'hemoglobin',
            'glucose': 'glucose',
            'chloride': 'chloride',
            'creatinine': 'creatinine',
            'aniongap': 'aniongap',
            'bicarbonate': 'bicarbonate',

            # other lab
            'hematocrit': 'hematocrit',

            # used
            'heart rate': 'heartrate',
            'respiratory rate': 'resprate',
            'temperature': 'tempc',
            'meanbp': 'meanbp',
            'gcs': 'gcs_min',
            'urineoutput': 'urineoutput',
            'sysbp': 'sysbp',
            'diasbp': 'diasbp',
            'spo2': 'spo2',
            'Magnesium': '',

            'C-reactive protein': '',
            'bands': 'bands',
            }

item_id_dict = {
            'C-reactive protein': '50889',
            'Magnesium': '50960', 
            }

def time_to_second(t):
    t = str(t).replace('"', '')
    t = time.mktime(time.strptime(t,'%Y-%m-%d %H:%M:%S'))
    return int(t)

def _safe_float(x):
    if x is None:
        return None
    x = str(x).strip()
    if x in ['', 'NA', 'NaN', 'nan', 'None', 'null', 'NULL']:
        return None
    try:
        return float(x)
    except Exception:
        return None

def _clip(v, vmin, vmax):
    if v is None:
        return None
    if v < vmin:
        return vmin
    if v > vmax:
        return vmax
    return v

def _mkdir_p(d):
    if d is None or str(d).strip() == '':
        raise ValueError('mkdir: empty path')
    try:
        os.makedirs(d)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(d):
            return
        raise

def preprocess_extracted_sepsis_timeseries():
    """
    Consume `sepsis_timeseries_hourly.csv` and write per-stay CSVs for TAME.

    Output format per file:
      header: time,<feature_1>,...,<feature_k>
      each row: <hour>,<value or empty>
    """
    extracted_path = os.path.join(args.mimic_dir, 'sepsis_timeseries_hourly.csv')
    if not os.path.exists(extracted_path):
        raise RuntimeError('Cannot find extracted sepsis file: {:s}'.format(extracted_path))

    # NOTE: tools/py_op.mkdir does not support absolute paths (it tries to mkdir('')).
    # Use our own mkdir -p implementation here.
    base_data_dir = os.path.join(os.path.abspath(args.data_dir), args.dataset)
    gt_dir = os.path.join(base_data_dir, 'train_groundtruth')
    wm_dir = os.path.join(base_data_dir, 'train_with_missing')
    _mkdir_p(base_data_dir)
    _mkdir_p(gt_dir)
    _mkdir_p(wm_dir)

    # Minimal multi-modal placeholders (TAME loader expects these for dataset == 'MIMIC').
    # We keep them empty, which is fine when `--use-mm 0` (default).
    ehr_list_path = os.path.join(base_data_dir, 'ehr_list.json')
    if not os.path.exists(ehr_list_path):
        py_op.mywritejson(ehr_list_path, [])

    # Column specs derived from `code/preprocessing/data.md`.
    # Each spec: input_col, output_col, range(min,max) or None, fill_strategy
    # - fill_strategy: 'keep' (keep missing as empty), 'zero' (missing->0)
    COL_SPECS = [
        ('heart_rate', 'heartrate', (30.0, 220.0), 'keep'),
        ('resp_rate', 'resprate', (5.0, 60.0), 'keep'),
        ('temperature', 'tempc', (32.0, 42.0), 'keep'),
        ('sbp', 'sysbp', (40.0, 250.0), 'keep'),
        ('dbp', 'diasbp', (30.0, 150.0), 'keep'),
        ('mbp', 'meanbp', (30.0, 200.0), 'keep'),
        ('spo2', 'spo2', (50.0, 100.0), 'keep'),
        ('gcs', 'gcs', (3.0, 15.0), 'keep'),
        ('urineoutput_sum', 'urineoutput', (0.0, 5000.0), 'keep'),
        ('wbc', 'wbc', (0.1, 150.0), 'keep'),
        ('hemoglobin', 'hemoglobin', (3.0, 20.0), 'keep'),
        ('hematocrit', 'hematocrit', (10.0, 60.0), 'keep'),
        ('platelet', 'platelet', (5.0, 1000.0), 'keep'),
        ('creatinine', 'creatinine', (0.1, 15.0), 'keep'),
        ('bun', 'bun', (1.0, 200.0), 'keep'),
        ('sodium', 'sodium', (110.0, 170.0), 'keep'),
        ('potassium', 'potassium', (1.5, 9.0), 'keep'),
        ('chloride', 'chloride', (70.0, 140.0), 'keep'),
        ('bicarbonate', 'bicarbonate', (5.0, 55.0), 'keep'),
        ('calcium', 'calcium', (4.0, 15.0), 'keep'),
        ('aniongap', 'aniongap', (5.0, 40.0), 'keep'),
        ('glucose_lab', 'glucose', (20.0, 800.0), 'keep'),
        ('bilirubin_total', 'bilirubin', (0.1, 40.0), 'keep'),
        ('inr', 'inr', (0.5, 15.0), 'keep'),
        ('pt', 'pt', (10.0, 150.0), 'keep'),
        ('ptt', 'ptt', (20.0, 150.0), 'keep'),
        ('lactate', 'lactate', (0.3, 25.0), 'keep'),
        ('magnesium', 'magnesium', (0.5, 5.0), 'keep'),

        # Respiratory function / SOFA-related continuous values
        ('pao2fio2ratio_novent', 'pao2fio2ratio_novent', (20.0, 700.0), 'keep'),
        ('pao2fio2ratio_vent', 'pao2fio2ratio_vent', (20.0, 700.0), 'keep'),

        # Vasopressor rates: fill missing with 0 (no infusion), do not time-fill.
        ('rate_epinephrine', 'rate_epinephrine', (0.0, 2.0), 'zero'),
        ('rate_norepinephrine', 'rate_norepinephrine', (0.0, 2.0), 'zero'),
        # NOTE: dopamine units may vary; keep the range conservative to avoid hard truncation.
        ('rate_dopamine', 'rate_dopamine', (0.0, 30.0), 'zero'),
        ('rate_dobutamine', 'rate_dobutamine', (0.0, 30.0), 'zero'),

        # Hourly minimums / rolling values
        ('meanbp_min', 'meanbp_min', (30.0, 200.0), 'keep'),
        ('gcs_min', 'gcs_min', (3.0, 15.0), 'keep'),
        ('uo_24hr', 'uo_24hr', (0.0, 20000.0), 'keep'),
        ('bilirubin_max', 'bilirubin_max', (0.1, 40.0), 'keep'),
        ('creatinine_max', 'creatinine_max', (0.1, 15.0), 'keep'),
        ('platelet_min', 'platelet_min', (5.0, 1000.0), 'keep'),

        # SOFA subscores (ordinal): keep as-is within [0,4]
        ('respiration', 'respiration', (0.0, 4.0), 'keep'),
        ('coagulation', 'coagulation', (0.0, 4.0), 'keep'),
        ('liver', 'liver', (0.0, 4.0), 'keep'),
        ('cardiovascular', 'cardiovascular', (0.0, 4.0), 'keep'),
        ('cns', 'cns', (0.0, 4.0), 'keep'),
        ('renal', 'renal', (0.0, 4.0), 'keep'),
        ('sofa_score', 'sofa_score', (0.0, 24.0), 'keep'),
        ('respiration_24hours', 'respiration_24hours', (0.0, 4.0), 'keep'),
        ('coagulation_24hours', 'coagulation_24hours', (0.0, 4.0), 'keep'),
        ('liver_24hours', 'liver_24hours', (0.0, 4.0), 'keep'),
        ('cardiovascular_24hours', 'cardiovascular_24hours', (0.0, 4.0), 'keep'),
        ('cns_24hours', 'cns_24hours', (0.0, 4.0), 'keep'),
        ('renal_24hours', 'renal_24hours', (0.0, 4.0), 'keep'),
        ('sofa_24hours', 'sofa_24hours', (0.0, 24.0), 'keep'),
    ]

    # Input cols to ignore (present in extracted CSV but not used as model features)
    IGNORE_INPUT_COLS = set([
        'stay_id', 'hour',
        'urineoutput_last',
        'albumin', 'bands', 'crp',
        'starttime', 'endtime',
        'urineoutput_24hr', 'uo_tm_24hr',
    ])

    # Validate expected columns exist.
    with open(extracted_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
    header = [h.strip() for h in header]
    header_set = set(header)
    for in_col, out_col, rr, strategy in COL_SPECS:
        if in_col not in header_set:
            raise RuntimeError('Missing required column in extracted CSV: {:s}'.format(in_col))
    if 'stay_id' not in header_set or 'hour' not in header_set:
        raise RuntimeError('Extracted CSV must contain stay_id and hour columns')

    # Prepare indices.
    stay_idx = header.index('stay_id')
    hour_idx = header.index('hour')
    in_indices = [header.index(in_col) for (in_col, _, _, _) in COL_SPECS]
    out_header = ['time'] + [out_col for (_, out_col, _, _) in COL_SPECS]
    out_header_line = ','.join(out_header) + '\n'

    # Stream-split per stay_id (file is sorted by stay_id then hour).
    n_written = 0
    current_stay = None
    current_lines = None  # buffer rows for current stay to enable with_missing generation

    def _flush_one_stay(stay_id, lines):
        """
        Write groundtruth CSV, with_missing CSV, and an empty multimodal JSON for one stay.
        `lines` includes header as first line already.
        """
        if stay_id is None or lines is None:
            return
        # Basic filtering to avoid ultra-sparse stays
        if len(lines) < 6:  # header + >=5 rows
            return
        gt_path = os.path.join(gt_dir, '{:s}.csv'.format(stay_id))
        wm_path = os.path.join(wm_dir, '{:s}.csv'.format(stay_id))

        # Write groundtruth
        with open(gt_path, 'w') as wf:
            wf.write(''.join(lines))

        # Create with_missing by masking one observed value per feature (excluding first/last valid).
        # Mirrors the original `generate_lab_missing_values` logic.
        rows = [ln.strip().split(',') for ln in lines]
        feat_list = rows[0]
        valid = np.zeros((len(rows), len(rows[0])), dtype=np.int64)
        for i in range(1, len(rows)):
            for j in range(len(rows[0])):
                if rows[i][j] not in ['', 'NA']:
                    valid[i, j] = 1
        valid[0] = 0
        for j in range(1, valid.shape[1]):
            idxs = np.where(valid[:, j] > 0)[0]
            idxs = list(sorted(idxs))
            if len(idxs) > 2:
                idxs = idxs[1:-1]
                random.shuffle(idxs)
                rows[idxs[0]][j] = ''
        with open(wm_path, 'w') as wf:
            wf.write('\n'.join([','.join(r) for r in rows]) + '\n')

        # Empty multimodal json (required by loader for dataset == 'MIMIC')
        json_path = os.path.join(gt_dir, '{:s}.json'.format(stay_id))
        if not os.path.exists(json_path):
            py_op.mywritejson(json_path, {'drug': {}, 'icd_demo': []})

    with open(extracted_path, 'r') as f:
        reader = csv.reader(f)
        header_in = next(reader)  # skip header
        for row in tqdm(reader):
            if len(row) == 0:
                continue
            stay_id = row[stay_idx].strip()
            hour = _safe_float(row[hour_idx])
            if stay_id in ['', 'NA'] or hour is None:
                continue
            hour = int(hour)
            # Keep same time window as original codebase
            if hour < -24 or hour >= 500:
                continue

            if current_stay is None:
                current_stay = stay_id
                current_lines = [out_header_line]
            elif stay_id != current_stay:
                _flush_one_stay(current_stay, current_lines)
                n_written += 1
                current_stay = stay_id
                current_lines = [out_header_line]

            out_vals = [str(hour)]
            for (idx, spec) in zip(in_indices, COL_SPECS):
                in_col, out_col, rr, strategy = spec
                v = _safe_float(row[idx])
                if v is None:
                    if strategy == 'zero':
                        v = 0.0
                    else:
                        out_vals.append('')
                        continue
                if rr is not None:
                    v = _clip(v, rr[0], rr[1])
                # keep consistent formatting
                if float(int(v)) == float(v):
                    out_vals.append(str(int(v)))
                else:
                    out_vals.append('{:0.4f}'.format(float(v)))
            current_lines.append(','.join(out_vals) + '\n')

    _flush_one_stay(current_stay, current_lines)
    print('Done. Wrote per-stay CSVs to:')
    print('  groundtruth:', gt_dir)
    print('  with_missing:', wm_dir)

def select_records_of_variables_not_in_pivoted():
    count_dict = { v:0 for v in item_id_dict.values() }
    hadm_time_dict = py_op.myreadjson(os.path.join(args.data_dir, args.dataset, 'hadm_time_dict.json' ))
    wf = open(os.path.join(args.mimic_dir, 'sepsis_lab.csv'), 'w')
    for i_line, line in enumerate(open(os.path.join(args.mimic_dir, 'LABEVENTS.csv'))):
        if i_line:
            line_data = line.split(',')
            if len(line_data) == 0:
                continue
            hadm_id, item_id, ctime = line_data[2:5]
            value = line_data[5]
            if item_id in count_dict and hadm_id in hadm_time_dict:
                # print(line)
                if len(line_data) != 9:
                    print(line)
                # assert len(line_data) == 9
                count_dict[item_id] += 1
                wf.write(line)
        else:
            wf.write(line)
            continue
        if i_line % 10000 == 0:
            print(i_line)
    wf.close()



def generate_variables_not_in_pivoted():
    assert args.dataset == 'MIMIC'
    id_item_dict = { v:k for k,v in item_id_dict.items() }
    head = sorted(item_id_dict)
    count_dict = { v:0 for v in item_id_dict.values() }
    wf = open(os.path.join(args.mimic_dir, 'pivoted_add.csv'), 'w')
    wf.write(','.join(['hadm_id', 'charttime'] + head) + '\n')
    for i_line, line in enumerate(open(os.path.join(args.mimic_dir, 'sepsis_lab.csv'))):
        if i_line:
            line_data = py_op.csv_split(line)
            hadm_id, item_id, ctime = line_data[2:5]
            value = line_data[6]
            try:
                value = float(value)
                index = head.index(id_item_dict[item_id])
                new_line = [hadm_id, ctime] + ['' for _ in range(index)] + [str(value)] + ['' for _ in range(index, len(head)-1)]
                new_line = ','.join(new_line) + '\n'
                wf.write(new_line)
            except:
                continue
            count_dict[item_id] += 1
            last_time = ctime
        else:
            print(line)
    print(count_dict)

def merge_pivoted_data(csv_list):
    name_list = ['hadm_id', 'charttime']
    for k,v in variable_map_dict.items():
        if k not in ['age', 'gender']:
            if len(v):
                name_list.append(v)
            elif k in item_id_dict:
                name_list.append(k)
    name_index_dict = { name:id for id,name in enumerate(name_list) }

    hadm_time_dict = py_op.myreadjson(os.path.join(args.data_dir, args.dataset, 'hadm_time_dict.json' ))
    icu_hadm_dict = py_op.myreadjson(os.path.join(args.data_dir, args.dataset, 'icu_hadm_dict.json' ))
    merge_dir = os.path.join(args.data_dir, args.dataset, 'merge_pivoted')
    os.system('rm -r ' + merge_dir)
    os.system('mkdir ' + merge_dir)
    pivoted_dir =  os.path.join(args.result_dir, 'mimic/pivoted_sofa')
    py_op.mkdir(pivoted_dir)

    for fi in csv_list:
        print(fi)
        for i_line, line in enumerate(open(os.path.join(args.mimic_dir, fi))):
            if i_line:
                line_data = line.strip().split(',')
                if len(line_data) <= 0:
                    continue
                line_dict = dict()
                for iv, v in enumerate(line_data):
                    if len(v.strip()):
                        name = head[iv]
                        line_dict[name] = v

                if fi == 'pivoted_sofa.csv':
                    icu_id = line_dict.get('icustay_id', 'xxx')
                    if icu_id not in icu_hadm_dict:
                        continue
                    hadm_id = str(icu_hadm_dict[icu_id])
                    line_dict['hadm_id'] = hadm_id
                    line_dict['charttime'] = line_dict['starttime']


                hadm_id = line_dict.get('hadm_id', 'xxx')
                if hadm_id not in hadm_time_dict:
                    continue
                hadm_time = time_to_second(hadm_time_dict[hadm_id])
                now_time = time_to_second(line_dict['charttime'])
                delta_hour = int((now_time - hadm_time) / 3600)
                line_dict['charttime'] = str(delta_hour)

                if fi == 'pivoted_sofa.csv':
                    sofa_file = os.path.join(pivoted_dir, hadm_id + '.csv')
                    if not os.path.exists(sofa_file):
                        with open(sofa_file, 'w') as f:
                            f.write(sofa_head)
                    wf = open(sofa_file, 'a')
                    sofa_line = [str(delta_hour)] + line.split(',')[4:]
                    wf.write(','.join(sofa_line))
                    wf.close()


                assert 'hadm_id' in line_dict
                assert 'charttime' in line_dict
                new_line = []
                for name in name_list:
                    new_line.append(line_dict.get(name, ''))
                new_line = ','.join(new_line) + '\n'
                hadm_file = os.path.join(merge_dir, hadm_id + '.csv')
                if not os.path.exists(hadm_file):
                    with open(hadm_file, 'w') as f:
                        f.write(','.join(name_list) + '\n')
                wf = open(hadm_file, 'a')
                wf.write(new_line)
                wf.close()

            else:
                if fi == 'pivoted_sofa.csv':
                    sofa_head = ','.join(['time'] + line.replace('"', '').split(',')[4:])
                # "icustay_id","hr","starttime","endtime","pao2fio2ratio_novent","pao2fio2ratio_vent","rate_epinephrine","rate_norepinephrine","rate_dopamine","rate_dobutamine","meanbp_min","gcs_min","urineoutput","bilirubin_max","creatinine_max","platelet_min","respiration","coagulation","liver","cardiovascular","cns","renal","respiration_24hours","coagulation_24hours","liver_24hours","cardiovascular_24hours","cns_24hours","renal_24hours","sofa_24hours"
                
                
                head = line.replace('"', '').strip().split(',')
                head = [h.strip() for h in head]
                # print(line)
                for h in head:
                    if h not in  name_index_dict:
                        print(h)

def sort_pivoted_data():
    sort_dir = os.path.join(args.data_dir, args.dataset, 'sort_pivoted')
    os.system('rm -r ' + sort_dir)
    os.system('mkdir ' + sort_dir)
    merge_dir = os.path.join(args.data_dir, args.dataset, 'merge_pivoted')

    for i_fi, fi in enumerate(tqdm(os.listdir(merge_dir))):
        wf = open(os.path.join(sort_dir, fi), 'w')
        time_line_dict = dict()
        for i_line, line in enumerate(open(os.path.join(merge_dir, fi))):
            if i_line:
                line_data = line.strip().split(',')
                delta = 3
                ctime = delta * int(int(line_data[1]) / delta)
                if ctime not in time_line_dict:
                    time_line_dict[ctime] = []
                time_line_dict[ctime].append(line_data)
            else:
                line_data = line.split(',')[1:]
                line_data[0] = 'time'
                wf.write(','.join(line_data))
        for t in sorted(time_line_dict):
            line_list = time_line_dict[t]
            new_line = line_list[0]
            for line_data in line_list[1:]:
                for iv, v in enumerate(line_data):
                    if len(v.strip()):
                        new_line[iv] = v
            new_line = ','.join(new_line[1:]) + '\n'
            wf.write(new_line)
        wf.close()
    py_op.mkdir('../../data/MIMIC/train_groundtruth')
    py_op.mkdir('../../data/MIMIC/train_with_missing')
    os.system('rm ../../data/MIMIC/train_groundtruth/*.csv')
    os.system('cp ../../data/MIMIC/sort_pivoted/* ../../data/MIMIC/train_groundtruth/')

def generate_icu_mortality_dict(icustay_id_list):
    icu_mortality_dict = dict()
    for i_line, line in enumerate(open(os.path.join(args.mimic_dir, 'sepsis_mortality.csv'))):
        if i_line:
            if i_line % 10000 == 0:
                print(i_line)
            line_data = line.strip().split(',')
            icustay_id = line_data[0]
            icu_mortality_dict[icustay_id] = int(line_data[-1])
    py_op.mywritejson(os.path.join(args.data_dir, 'icu_mortality_dict.json'), icu_mortality_dict)


def generate_lab_missing_values():
    lab_files = glob(os.path.join(args.data_dir, args.dataset, 'train_groundtruth/*.csv'))
    os.system('rm -r {:s}/*'.format(os.path.join(args.data_dir, args.dataset, 'train_with_missing')))
    feat_count_dict = dict()
    line_count_dict = dict()
    n_full = 0
    for i_fi, fi in enumerate(tqdm(lab_files)):
        file_data = []
        valid_list = []
        last_data = [-10000]
        for i_line, line in enumerate(open(fi)):
            if i_line:
                data = line.strip().split(',')
                # print(data)
                # print(fi)
                # assert(int(data[0])) > -200
                # assert(int(data[0])) < 800
                if int(data[0]) < -24 or int(data[0]) >= 500: 
                    continue
                assert int(data[0]) > -200
                valid = []
                for i in range(len(data)):
                    feat_count_dict[feat_list[i]][0] += 1
                    if data[i] in ['', 'NA']:
                        feat_count_dict[feat_list[i]][1] += 1
                        valid.append(0)
                    else:
                        valid.append(1)
                        vector[i] = 1

                if data[0] == last_data[0]:
                    for iv in range(len(data)):
                        if valid[iv]:
                            last_valid[iv] = valid[iv]
                            last_data[iv] = data[iv]
                    valid_list[-1] = last_valid
                    file_data[-1] = last_data
                else:
                    valid_list.append(valid)
                    assert int(data[0]) < 700
                    assert int(data[0]) > - 200
                    file_data.append(data)
                    last_data = data
                    last_valid = valid
            else:
                feat_list = line.strip().split(',')
                vector = [0 for _ in feat_list]
                for feat in feat_list:
                    if feat not in feat_count_dict:
                        feat_count_dict[feat] = [0, 0]

                valid_list.append([1 for _ in feat_list])
                file_data.append(feat_list)
        line_count_dict[i_line] = line_count_dict.get(i_line, 0) + 1

        vs = [0 for _ in file_data[0]]
        for data in file_data[1:]:
            for iv, v in enumerate(data):
                if v.strip() not in ['', 'NA']:
                    vs[iv] += 1
        if np.min(vs) >= 2:
            n_full +=1
        # continue


        # if len(file_data)< 15 or np.min(vs) < 2:
        if len(file_data)< 5 or sorted(vector)[2] < 1:
            os.system('rm -r ' + fi)
            # os.system('rm -r ' + fi.replace('groundtruth', 'with_missing'))
            # print('rm -r ' + fi.replace('groundtruth', 'with_missing'))
        else:
            for data in file_data[1:]:
                assert int(data[0]) > -200
            # write groundtruth data
            x = [','.join(line) for line in file_data]
            x = '\n'.join(x)
            with open(fi, 'w') as f:
                f.write(x)

            valid_list = np.array(valid_list)
            valid_list[0] = 0
            for i in range(1, valid_list.shape[1]):
                valid = valid_list[:, i]
                indices = np.where(valid > 0)[0]
                indices = sorted(indices)
                if len(indices) > 2:
                    indices = indices[1:-1]
                    np.random.shuffle(indices)
                    file_data[indices[0]][i] = ''
            # write groundtruth data
            x = [','.join(line) for line in file_data]
            x = '\n'.join(x)
            with open(fi.replace('groundtruth', 'with_missing'), 'w') as f:
                f.write(x)
    print(n_full)


def main():
    # If the user points --mimic-dir to an extracted sepsis export folder,
    # we can preprocess directly without needing raw MIMIC-III tables.
    extracted_path = os.path.join(args.mimic_dir, 'sepsis_timeseries_hourly.csv')
    if os.path.exists(extracted_path):
        preprocess_extracted_sepsis_timeseries()
        return

    # Legacy MIMIC-III mode (requires raw MIMIC-III tables + pivoted_*.csv).
    csv_list = ['pivoted_sofa.csv', 'pivoted_add.csv', 'pivoted_lab.csv', 'pivoted_vital.csv']
    select_records_of_variables_not_in_pivoted()
    generate_variables_not_in_pivoted()
    merge_pivoted_data(csv_list)
    sort_pivoted_data()
    generate_lab_missing_values()



if __name__ == '__main__':
    main()
