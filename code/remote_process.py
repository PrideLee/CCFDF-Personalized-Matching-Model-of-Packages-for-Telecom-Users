import pandas as pd
import numpy as np
import random


def pro(file_dataframe, par_list):
    num = len(file_dataframe)
    for i in par_list:
        list_par = file_dataframe[i].tolist()
        quantile = np.percentile(list_par, (25, 75), interpolation='midpoint').tolist()
        IQR = quantile[1] - quantile[0]
        up_bound = quantile[1] + 1.5 * IQR
        down_bound = quantile[0] - 1.5 * IQR
        interval = up_bound - down_bound
        pre = []
        for j in range(num):
            val = list_par[j]
            val_up = val + val * 0.02 * interval
            val_down = val - val * 0.02 * interval
            interval_num = [k for k in list_par if val_down < k < val_up]
            pre.append(len(interval_num) / num)
        att_new = i + "_precentage"
        file_dataframe[att_new] = pre
    return file_dataframe


raw_data = pd.read_csv(r"/data/projects/CCFDF_18/data/combine_4_norm.csv", encoding="utf-8", low_memory=False)
par = ['month_traffic', 'contract_time', 'pay_times', 'pay_num', 'last_month_traffic', 'local_trafffic_month',
       'local_caller_time', 'service1_caller_time', 'service2_caller_time']
dataframe_sel = pro(raw_data, par)
dataframe_sel.to_csv(r"/data/projects/CCFDF_18/data/combine_4_precentage.csv")
