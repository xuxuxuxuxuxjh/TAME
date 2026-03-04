| 序号 | 变量代码 (您的数据) | 对应 TAME 变量 | 中文含义 | 类别 | 处理方式/建议范围 | 缺失率 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | stay_id | 无 | ICU入住ID | 标识符 | 排除 | - |
| 2 | hour | 无 | 时间点 | 时间索引 | 排除 | - |
| 3 | heart_rate | heartrate | 心率 | 生命体征 | [30-220] bpm; 时序补全 | 5% |
| 4 | resp_rate | resprate | 呼吸频率 | 生命体征 | [5-60] bpm; 时序补全 | 5% |
| 5 | temperature | tempc | 体温 | 生命体征 | [32-42] °C; 时序补全 | 68% |
| 6 | sbp | sysbp | 收缩压 | 生命体征 | [40-250] mmHg; 时序补全 | 8% |
| 7 | dbp | diasbp | 舒张压 | 生命体征 | [30-150] mmHg; 时序补全 | 8% |
| 8 | mbp | meanbp | 平均动脉压 | 生命体征 | [30-200] mmHg; 时序补全 | 8% |
| 9 | spo2 | spo2 | 血氧饱和度 | 生命体征 | [50-100] %; 时序补全 | 8% |
| 10 | gcs | gcs | 昏迷评分 | 生命体征 | [3-15]; 时序补全 | 70% |
| 11 | urineoutput_last | 无 | 最近1h尿量 | 临床指标 | [0-1000] mL; 排除，值同urineoutput_sum | - |
| 12 | urineoutput_sum | urineoutput | 累计尿量 | 临床指标 | [0-5000] mL; 时序补全 | 40% |
| 13 | wbc | wbc | 白细胞 | 血常规 | [0.1-150] 10^9/L; 时序补全 | 90% |
| 14 | hemoglobin | hemoglobin | 血红蛋白 | 血常规 | [3-20] g/dL; 时序补全 | 90% |
| 15 | hematocrit | hematocrit | 红细胞压积 | 血常规 | [10-60] %; 时序补全 | 90% |
| 16 | platelet | platelet | 血小板 | 血常规 | [5-1000] 10^9/L; 时序补全 | 90% |
| 17 | creatinine | creatinine | 肌酐 | 肾功能 | [0.1-15] mg/dL; 时序补全 | 90% |
| 18 | bun | bun | 尿素氮 | 肾功能 | [1-200] mg/dL; 时序补全 | 90% |
| 19 | sodium | sodium | 血钠 | 电解质 | [110-170] mmol/L; 时序补全 | 90% |
| 20 | potassium | potassium | 血钾 | 电解质 | [1.5-9.0] mmol/L; 时序补全 | 90% |
| 21 | chloride | chloride | 血氯 | 电解质 | [70-140] mmol/L; 时序补全 | 90% |
| 22 | bicarbonate | bicarbonate | 碳酸氢盐 | 酸碱平衡 | [5-55] mmol/L; 时序补全 | 90% |
| 23 | calcium | 无 | 血钙 | 电解质 | [4-15] mg/dL; 新增补全 | 90% |
| 24 | aniongap | aniongap | 阴离子间隙 | 酸碱平衡 | [5-40] mmol/L; 时序补全 | 90% |
| 25 | glucose_lab | glucose | 血糖 | 生化代谢 | [20-800] mg/dL; 时序补全 | 90% |
| 26 | albumin | 无 | 白蛋白 | 生化指标 | [1.0-5.5] g/dL; 排除，太少了 | 98% |
| 27 | bilirubin_total | bilirubin | 总胆红素 | 肝功能 | [0.1-40] mg/dL; 时序补全 | 96% |
| 28 | inr | inr | 凝血INR | 凝血功能 | [0.5-15]; 时序补全 | 92% |
| 29 | pt | pt | 凝血酶原时间 | 凝血功能 | [10-150] s; 时序补全 | 92% |
| 30 | ptt | ptt | 分凝血时间 | 凝血功能 | [20-150] s; 时序补全 | 92% |
| 31 | lactate | lactate | 乳酸 | 生化代谢 | [0.3-25] mmol/L; 时序补全 | 92% |
| 32 | bands | 无 | 杆状核中性粒 | 感染指标 | [0-60] %; 排除，太少了 | 99% |
| 33 | crp | 无 | C反应蛋白 | 炎症指标 | [0-400] mg/L; 排除，太少了 | 99% |
| 34 | magnesium | magnesium | 血镁 | 电解质 | [0.5-5.0] mg/dL; 时序补全 | 90% |
| 35 | starttime | 无 | 开始时间 | 时间 | 排除 | - |
| 36 | endtime | 无 | 结束时间 | 时间 | 排除 | - |
| 37 | pao2fio2ratio_novent | 无 | P/F比(非通气) | 呼吸功能 | [20-700]; 新增补全 | 99% |
| 38 | pao2fio2ratio_vent | 无 | P/F比(通气) | 呼吸功能 | [20-700]; 新增补全 | 94% |
| 39 | rate_epinephrine | 无 | 肾上腺素速率 | 药物 | [0-2.0]; 填0, Mask=1，不进行补全 | 97% |
| 40 | rate_norepinephrine | 无 | 去甲肾速率 | 药物 | [0-2.0]; 填0, Mask=1，不进行补全 | 85% |
| 41 | rate_dopamine | 无 | 多巴胺速率 | 药物 | [0-2.0]; 填0, Mask=1，不进行补全 | 99% |
| 42 | rate_dobutamine | 无 | 多巴酚丁胺速率 | 药物 | [0-2.0]; 填0, Mask=1，不进行补全 | 99% |
| 43 | meanbp_min | 无 | 最低平均动脉压 | 静态统计 | 时序补全 | 12% |
| 44 | gcs_min | 无 | 最低 GCS | 静态统计 | 时序补全 | 70% |
| 45 | urineoutput_24hr | 无 | 24h总尿量 | 静态统计 | 排除 | 40% |
| 46 | uo_tm_24hr | 无 | 24h尿量时长 | 静态统计 | 排除 | 40% |
| 47 | uo_24hr | 无 | 24h总尿量(校验) | 静态统计 | 时序补全 | 92% |
| 48 | bilirubin_max | 无 | 最高总胆红素 | 静态统计 | 时序补全 | 96% |
| 49 | creatinine_max | 无 | 最高肌酐 | 静态统计 | 时序补全 | 90% |
| 50 | platelet_min | 无 | 最低血小板 | 静态统计 | 时序补全 | 90% |
| 51-56 | respiration...renal | 无 | SOFA子项 | 时序评分 | [0-4]; 无需补全 | - |
| 57 | sofa_score | 无 | 每小时SOFA总分 | 时序评分 | [0-24]; 无需补全 | - |
| 58-63 | respiration_24h... | 无 | 24h各系统分值 | 静态统计 | [0-4]; 无需补全 | - |
| 64 | sofa_24hours | 无 | 24h总SOFA评分 | 静态统计 | [0-24]; 无需补全 | - |