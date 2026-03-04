#!/usr/bin/env python
# coding=utf-8


import sys
try:
    # python3 compatibility
    from importlib import reload  # noqa: F401
except Exception:
    pass
try:
    # python2 compatibility (no-op on python3)
    reload(sys)  # type: ignore[name-defined]
    sys.setdefaultencoding('utf8')  # type: ignore[attr-defined]
except Exception:
    pass

import os
import sys
import time
import datetime
import numpy as np
try:
    # optional; not required for imputation training
    from sklearn import metrics  # noqa: F401
except Exception:
    pass
import random
import json
from glob import glob
from collections import OrderedDict
from tqdm import tqdm


import torch
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader

import data_loader
from models import tame
import myloss
import function

sys.path.append('../tools')
import parse, py_op

args = parse.args
args.hidden_size = args.rnn_size = args.embed_size 
if torch.cuda.is_available():
    args.gpu = 1
else:
    args.gpu = 0
if args.model != 'tame':
    args.use_ve = 0
    args.use_mm = 0
    args.use_ta = 0
if args.use_ve == 0:
    args.value_embedding = 'no'
print('epochs,', args.epochs)

def _safe_makedirs(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def _write_loss_csv(csv_file: str, rows):
    """
    rows: iterable of dicts with keys: epoch, train_loss, valid_loss
    """
    header = "epoch,train_loss,valid_loss\n"
    with open(csv_file, "w") as f:
        f.write(header)
        for r in rows:
            e = r.get("epoch", "")
            tl = r.get("train_loss", "")
            vl = r.get("valid_loss", "")
            f.write(f"{e},{tl},{vl}\n")

def _try_plot_loss_png(png_file: str, rows):
    """
    Best-effort plotting; if matplotlib is unavailable, do not fail training.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("WARN: matplotlib not available; skipping loss plot:", e)
        return

    xs = [r["epoch"] for r in rows]
    ys_t = [r.get("train_loss") for r in rows]
    ys_v = [r.get("valid_loss") for r in rows]

    plt.figure(figsize=(8, 4.5))
    if any(v is not None for v in ys_t):
        plt.plot(xs, ys_t, label="train")
    if any(v is not None for v in ys_v):
        plt.plot(xs, ys_v, label="valid")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("TAME loss curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_file, dpi=160)
    plt.close()

def _cuda(tensor, is_tensor=True):
    if args.gpu:
        # `async=True` was removed; use non_blocking for pinned-memory transfers.
        if hasattr(tensor, 'cuda'):
            if is_tensor:
                return tensor.cuda(non_blocking=True)
            return tensor.cuda()
        return tensor
    else:
        return tensor

def get_lr(epoch):
    lr = args.lr
    return lr

    if epoch <= args.epochs * 0.5:
        lr = args.lr
    elif epoch <= args.epochs * 0.75:
        lr = 0.1 * args.lr
    elif epoch <= args.epochs * 0.9:
        lr = 0.01 * args.lr
    else:
        lr = 0.001 * args.lr
    return lr

def index_value(data):
    '''
    map data to index and value
    '''
    if args.use_ve == 0:
        return _cuda(data)  # [bs, n_visit, n_feat]
    data_np = data.detach().cpu().numpy()
    index = data_np // (args.split_num + 1)
    value = data_np % (args.split_num + 1)
    index_t = _cuda(torch.from_numpy(index.astype(np.int64)))
    value_t = _cuda(torch.from_numpy(value.astype(np.int64)))
    return [index_t, value_t]

def train_eval(data_loader, net, loss, epoch, optimizer, best_metric, phase='train'):
    print(phase)
    lr = get_lr(epoch)
    if phase == 'train':
        net.train()
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        net.eval()

    loss_list, pred_list, label_list, mask_list = [], [], [], []
    feature_mm_dict = py_op.myreadjson(os.path.join(args.file_dir, args.dataset + '_feature_mm_dict.json'))
    for b, data_list in enumerate(tqdm(data_loader)):
        data, label, mask, files = data_list[:4]
        data = index_value(data)
        if args.model == 'tame':
            pre_input, pre_time, post_input, post_time, dd_list = data_list [4:9]
            pre_input = index_value(pre_input)
            post_input = index_value(post_input)
            pre_time = _cuda(pre_time)
            post_time = _cuda(post_time)
            dd_list = _cuda(dd_list)
            neib = [pre_input, pre_time, post_input, post_time]

        label = _cuda(label)
        mask = _cuda(mask)
        if args.dataset in ['MIMIC'] and args.model == 'tame' and args.use_mm:
            output = net(data, neib=neib, dd=dd_list, mask=mask) # [bs, 1]
        elif args.model == 'tame' and args.use_ta:
            output = net(data, neib=neib, mask=mask) # [bs, 1]
        else:
            output = net(data, mask=mask) # [bs, 1]

        if phase == 'test':
            # Write outputs to a dedicated directory (never overwrite original CSVs).
            folder = args.impute_dir
            if not folder:
                run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                folder = os.path.join(args.result_dir, args.dataset, 'imputation_result', run_id)
            if not os.path.exists(folder):
                os.makedirs(folder)
            output_data = output.detach().cpu().numpy()
            mask_data = mask.detach().cpu().numpy().max(2)
            for (icu_data, icu_mask, icu_file) in zip(output_data, mask_data, files):
                icu_file = os.path.join(folder, icu_file.split('/')[-1].replace('.csv', '.npy'))
                np.save(icu_file, icu_data)
                if args.dataset == 'MIMIC':
                    with open(os.path.join(args.data_dir, args.dataset, 'train_groundtruth', icu_file.split('/')[-1].replace('.npy', '.csv'))) as f:
                        init_data = f.read().strip().split('\n')
                    # print(icu_file)
                    wf = open(icu_file.replace('.npy', '.csv'), 'w')
                    wf.write(init_data[0] + '\n')
                    item_list = init_data[0].strip().split(',')
                    if len(init_data) <= args.n_visit:
                        try:
                            assert len(init_data) == (icu_mask >= 0).sum() + 1
                        except:
                            pass
                            # print(len(init_data))
                            # print(sum(icu_mask >= 0))
                            # print(icu_file)
                    for init_line, out_line in zip(init_data[1:], icu_data):
                        init_line = init_line.strip().split(',')
                        new_line = [init_line[0]]
                        # assert len(init_line) == len(out_line) + 1
                        for item, iv, ov in zip(item_list[1:], init_line[1:], out_line):
                            if iv.strip() not in ['', 'NA']:
                                new_line.append('{:4.4f}'.format(float(iv.strip())))
                            else:
                                minv, maxv = feature_mm_dict[item]
                                ov = ov * (maxv - minv) + minv
                                new_line.append('{:4.4f}'.format(ov))
                        new_line = ','.join(new_line)
                        wf.write(new_line + '\n')
                    wf.close()


        loss_output = loss(output, label, mask)
        pred_list.append(output.detach().cpu().numpy())
        loss_list.append(loss_output.detach().cpu().numpy())
        label_list.append(label.detach().cpu().numpy())
        mask_list.append(mask.detach().cpu().numpy())

        if phase == 'train':
            optimizer.zero_grad()
            loss_output.backward()
            optimizer.step()


    pred = np.concatenate(pred_list, 0)
    label = np.concatenate(label_list, 0)
    mask = np.concatenate(mask_list, 0)
    metric_list = function.compute_nRMSE(pred, label, mask)
    avg_loss = np.mean(loss_list)

    print('\n%s Epoch %03d (lr %.5f)' % (phase, epoch, lr))
    print('loss: {:3.4f} \t'.format(avg_loss))
    print('metric: {:s}'.format('\t'.join(['{:3.4f}'.format(m) for m in metric_list[:2]])))


    metric = metric_list[0]
    if phase == 'valid' and (best_metric[0] == 0 or best_metric[0] > metric):
        best_metric = [metric, epoch]
        # Save best checkpoint into a run-specific folder (no overwrite across runs),
        # and also save to the legacy global folder for compatibility with existing scripts.
        run_folder = getattr(args, "ckpt_dir", None)
        global_folder = getattr(args, "ckpt_global_dir", None)

        if run_folder:
            _safe_makedirs(run_folder)
            function.save_model(
                {'args': args, 'model': net, 'epoch': epoch, 'best_metric': best_metric},
                name='best.ckpt',
                folder=run_folder,
            )
            print("saved best.ckpt to:", os.path.join(run_folder, 'best.ckpt'))

        # Always keep a global best.ckpt (may be overwritten by subsequent runs)
        if global_folder:
            _safe_makedirs(global_folder)
            function.save_model(
                {'args': args, 'model': net, 'epoch': epoch, 'best_metric': best_metric},
                name='best.ckpt',
                folder=global_folder,
            )
            print("updated global best.ckpt:", os.path.join(global_folder, 'best.ckpt'))
        else:
            # fallback to original default behavior
            function.save_model({'args': args, 'model': net, 'epoch': epoch, 'best_metric': best_metric})
    metric_list = metric_list[2:]
    name_list = args.name_list
    assert len(metric_list) == len(name_list) * 2
    s = args.model
    for i in range(len(metric_list)//2):
        name = name_list[i] + ''.join(['.' for _ in range(40 - len(name_list[i]))])
        print('{:s}{:3.4f}......{:3.4f}'.format(name, metric_list[2*i], metric_list[2*i+1]))
        s = s+ '  {:3.4f}'.format(metric_list[2*i])
    if phase != 'train':
        print('\t\t\t\t best epoch: {:d}     best MSE on missing value: {:3.4f} \t'.format(best_metric[1], best_metric[0])) 
        print(s)
    return best_metric, float(avg_loss)


def main():

    assert args.dataset in ['DACMI', 'MIMIC']
    if args.dataset == 'MIMIC':
        args.n_ehr = len(py_op.myreadjson(os.path.join(args.data_dir, args.dataset, 'ehr_list.json')))
    args.name_list = py_op.myreadjson(os.path.join(args.file_dir, args.dataset+'_feature_list.json'))[1:]
    args.output_size = len(args.name_list)
    files = sorted(glob(os.path.join(args.data_dir, args.dataset, 'train_with_missing/*.csv')))
    data_splits = py_op.myreadjson(os.path.join(args.file_dir, args.dataset + '_splits.json'))
    train_files = [f for idx in [0, 1, 2, 3, 4, 5, 6] for f in data_splits[idx]]
    valid_files = [f for idx in [7] for f in data_splits[idx]]
    test_files = [f for idx in [8, 9] for f in data_splits[idx]]
    if args.phase == 'test':
        train_phase, valid_phase, test_phase, train_shuffle = 'test', 'test', 'test', False
    else:
        train_phase, valid_phase, test_phase, train_shuffle = 'train', 'valid', 'test', True
    train_dataset = data_loader.DataBowl(args, train_files, phase=train_phase)
    valid_dataset = data_loader.DataBowl(args, valid_files, phase=valid_phase)
    test_dataset = data_loader.DataBowl(args, test_files, phase=test_phase)
    pin = True if args.gpu else False
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=train_shuffle, num_workers=args.workers, pin_memory=pin)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin)
    args.vocab_size = (args.output_size + 2) * (1 + args.split_num) + 5

    if args.model == 'tame':
        net = tame.AutoEncoder(args)
    loss = myloss.MSELoss(args)

    net = _cuda(net, 0)
    loss = _cuda(loss, 0)

    best_metric= [0,0]
    start_epoch = 0

    if args.resume:
        p_dict = {'model': net}
        function.load_model(p_dict, args.resume)
        best_metric = p_dict['best_metric']
        start_epoch = p_dict['epoch'] + 1

    parameters_all = []
    for p in net.parameters():
        parameters_all.append(p)

    if args.phase == 'train':
        # Where to save loss curve artifacts
        run_id = os.environ.get("SLURM_JOB_ID") or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        loss_dir = os.path.join(args.result_dir, args.dataset, "loss_curve", f"{args.model}_{run_id}")
        _safe_makedirs(loss_dir)
        loss_csv = os.path.join(loss_dir, "loss.csv")
        loss_png = os.path.join(loss_dir, "loss.png")
        print("loss artifacts dir:", loss_dir)

        # Where to save run-specific checkpoints (avoid overwriting between runs)
        args.ckpt_dir = os.path.join(args.result_dir, args.dataset, "checkpoints", f"{args.model}_{run_id}")
        # Also keep legacy/global ckpt location for backward compatibility
        args.ckpt_global_dir = os.path.join(args.data_dir, args.dataset, "models")
        _safe_makedirs(args.ckpt_dir)
        _safe_makedirs(args.ckpt_global_dir)
        print("ckpt dir (run)     :", args.ckpt_dir)
        print("ckpt dir (global)  :", args.ckpt_global_dir)

        try:
            optimizer = torch.optim.Adam(parameters_all, args.lr)
        except ImportError as e:
            msg = str(e)
            # PyTorch 2.5+ may import torch._dynamo (which imports sympy) during optimizer creation.
            # If sympy is present but its dependency mpmath is missing, the error is confusing.
            if ("mpmath" in msg) or ("SymPy now depends on mpmath" in msg):
                raise SystemExit(
                    "ERROR: Missing Python package 'mpmath' required by sympy (imported by PyTorch torch._dynamo).\n"
                    "Fix once in your conda env, e.g.:\n"
                    "  conda activate postgre\n"
                    "  conda install -c conda-forge -y mpmath\n"
                    "or:\n"
                    "  pip install mpmath\n"
                ) from None
            raise
        loss_rows = []
        for epoch in range(start_epoch, args.epochs):
            print('start epoch :', epoch)
            best_metric, train_loss = train_eval(train_loader, net, loss, epoch, optimizer, best_metric, phase='train')
            best_metric, valid_loss = train_eval(valid_loader, net, loss, epoch, optimizer, best_metric, phase='valid')

            loss_rows.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
            # Write CSV every epoch (cheap + robust), and plot best-effort.
            _write_loss_csv(loss_csv, loss_rows)
            _try_plot_loss_png(loss_png, loss_rows)
        print('best metric', best_metric)

    elif args.phase == 'test':
        # Set output folder (do not delete anything).
        if not args.impute_dir:
            run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            args.impute_dir = os.path.join(args.result_dir, args.dataset, 'imputation_result', run_id)
        if not os.path.exists(args.impute_dir):
            os.makedirs(args.impute_dir)
        print('imputation output dir:', args.impute_dir)

        optimizer = None
        train_eval(train_loader, net, loss, 0, optimizer, best_metric, 'test')
        train_eval(valid_loader, net, loss, 0, optimizer, best_metric, 'test')
        train_eval(test_loader, net, loss, 0, optimizer, best_metric, 'test')

if __name__ == '__main__':
    main()
