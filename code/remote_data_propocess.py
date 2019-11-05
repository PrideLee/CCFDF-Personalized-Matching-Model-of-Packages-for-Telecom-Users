import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


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


def Min_Max(normalization_parameter, file_dataframe):
    """
    This function is used to data standardization.

    Parameters
    ----------
    normalization_parameter : list
    The attribute we want to normalize.
    file_dataframe : DataFrame
    The DataFrame consists of key attributes.

    Returen
    -------
    file_dataframe : DataFrame
    The DataFrame finished normalized.
    """
    normalization_parameter_result = []
    for i in normalization_parameter:
        par = file_dataframe[i].tolist()
        Max_par = max(par)
        Min_par = min(par)
        div = Max_par - Min_par
        if div == 0:
            if Max_par == 0:
                norm = par
            else:
                norm = [j / Max_par for j in par]
        else:
            norm = [round((j - Min_par) / div, 6) for j in par]
        normalization_parameter_result.append(norm)
    for j in range(len(normalization_parameter)):
        norm_name = normalization_parameter[j] + '_norm'
        file_dataframe[norm_name] = normalization_parameter_result[j]
    # file_dataframe_norm = file_dataframe.drop(normalization_parameter, axis=1)
    return file_dataframe


def pre(dataframe, paralist):
    # dataframe only include ther data of current_service = 12, 13, 14.
    service_type = dataframe["current_service"].tolist()
    num = len(service_type)
    list_12 = []
    list_13 = []
    list_14 = []
    for i in range(num):
        if service_type[i] == 12:
            list_12.append(i)
        if service_type[i] == 13:
            list_13.append(i)
        if service_type[i] == 14:
            list_14.append(i)
    dataframe_12 = dataframe.iloc[list_12]
    dataframe_13 = dataframe.iloc[list_13]
    dataframe_14 = dataframe.iloc[list_14]
    paralist_12 = []
    paralist_13 = []
    paralist_14 = []
    for i in paralist:
        paralist_12.append(sorted(dataframe_12[i].tolist()))
        paralist_13.append(sorted(dataframe_13[i].tolist()))
        paralist_14.append(sorted(dataframe_14[i].tolist()))
    para_num = len(paralist)
    num_12 = len(paralist_12[0])
    bound_12 = max(round(num_12 * 0.001), 5)
    para_12_bound = [round(paralist_12[j][num_12 - 1] - paralist_12[j][0]) for j in range(para_num)]
    pre_12 = [[] for i in range(para_num)]
    num_13 = len(paralist_13[0])
    bound_13 = max(round(num_13 * 0.001), 5)
    para_13_bound = [round(paralist_13[j][num_13 - 1] - paralist_13[j][0]) for j in range(para_num)]
    pre_13 = [[] for i in range(para_num)]
    num_14 = len(paralist_14[0])
    bound_14 = max(round(num_14 * 0.001), 5)
    para_14_bound = [round(paralist_14[j][num_14 - 1] - paralist_14[j][0]) for j in range(para_num)]
    pre_14 = [[] for i in range(para_num)]
    for i in range(num):
        for j in range(para_num):
            temp = dataframe.iloc[i][paralist[j]]
            for k in range(num_12):
                if temp < paralist_12[j][k]:
                    paralist_12[j].insert(k, temp)
                    break
            up = min(k + bound_12, -1)
            down = max(k - bound_12, 0)
            pre_12[j].append(round((paralist_12[j][up] - paralist_12[j][down]) / para_12_bound[j], 10))

            for k in range(num_13):
                if temp < paralist_13[j][k]:
                    paralist_13[j].insert(k, temp)
                    break
            up = min(k + bound_13, num_13 - 1)
            down = max(k - bound_13, 0)
            pre_13[j].append(round((paralist_13[j][up] - paralist_13[j][down]) / para_13_bound[j], 10))

            for k in range(num_14):
                if temp < paralist_14[j][k]:
                    paralist_14[j].insert(k, temp)
                    break

            up = min(k + bound_14, num_14 - 1)
            down = max(k - bound_14, 0)
            pre_14[j].append(round((paralist_14[j][up] - paralist_14[j][down]) / para_14_bound[j], 10))

    for j in range(para_num):
        rank_att = []
        for i in range(num):
            a = pre_12[j][i]
            b = pre_13[j][i]
            c = pre_14[j][i]
            temp_list = [a, b, c]
            list_sort = sorted(temp_list)
            rank = [list_sort.index(k) for k in temp_list]
            rank_att.append(rank[0] * 100 + rank[1] + rank[2])
        dataframe[paralist[j] + "_probability"] = rank_att
    return dataframe, pre_12, pre_13, pre_14


def bbox(raw_data, att):
    y0 = raw_data[raw_data["current_service"] == 12][att].tolist()
    y1 = raw_data[raw_data["current_service"] == 13][att].tolist()
    y2 = raw_data[raw_data["current_service"] == 14][att].tolist()
    y = np.transpose(np.array([y0, y1, y2]))
    labels = ["service_type_1", "service_type_3", "service_type_4"]
    plt.boxplot(y, labels=labels, sym='o')
    plt.grid(True)
    plt.show()


#
# raw_data = pd.read_csv(r"/data/projects/CCFDF_18/data/class_2_sup_add_0.csv", encoding="utf-8", low_memory=False)
#
# par = ['month_traffic', 'contract_time', 'pay_times', 'pay_num', 'last_month_traffic', 'local_trafffic_month',
#        'local_caller_time', 'service1_caller_time', 'service2_caller_time']
# dataframe_sel = pro(raw_data, par)
# dataframe_sel.to_csv(r"/data/projects/CCFDF_18/data/class_2_sup_add_1.csv")
# select_type_dataframe = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_sup_add_0_correct_exclude.csv",
#                                     encoding="utf-8", low_memory=False)
# para_list = ['online_time', 'local_trafffic_month', 'service2_caller_time', 'age', 'pay_num',
#              'fee_mean', 'fee_mean_2', 'fee_fluctuate', 'month_traffic', 'contract_time', 'pay_times',
#              'last_month_traffic', 'local_trafffic_month', 'local_caller_time',
#              'service1_caller_time', 'fee_std']
# new_dataframe = Min_Max(para_list, select_type_dataframe)
# new_dataframe.to_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_sup_add_new.csv")
# raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_sup_add_new.csv", encoding="utf-8", low_memory=False)
# raw_data_type = raw_data["current_service"].tolist()
# num_to = len(raw_data_type)
# del_type = [1, 2, 3]
# save_index = [i for i in range(num_to) if raw_data_type[i] not in del_type]
# data_save = raw_data.iloc[save_index]
# data_save.to_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_laste.csv",)


# num = len(select_type_dataframe)
# temp = select_type_dataframe["local_trafffic_month"].tolist()
# error_1 = [i for i in range(num) if temp[i] > 10 ** 7]
# select_type_dataframe["local_trafffic_month"][error_1] = 0
# temp = select_type_dataframe["pay_num"].tolist()
# error_2 = [i for i in range(num) if temp[i] > 30000]
# select_type_dataframe["pay_num"][error_2] = 0
# temp = select_type_dataframe["fee_mean"].tolist()
# error_3 = [i for i in range(num) if temp[i] > 15000]
# select_type_dataframe["fee_mean"][error_3] = 0
# temp = select_type_dataframe["fee_mean_2"].tolist()
# error_4 = [i for i in range(num) if (temp[i] > 10000) | (temp[i] < -10000)]
# select_type_dataframe["fee_mean_2"][error_4] = 0
# temp = select_type_dataframe["month_traffic"].tolist()
# error_5 = [i for i in range(num) if temp[i] > 10000]
# select_type_dataframe["month_traffic"][error_5] = 0
# temp = select_type_dataframe["pay_times"].tolist()
# error_6 = [i for i in range(num) if temp[i] > 10]
# select_type_dataframe["pay_times"][error_6] = 0
# temp = select_type_dataframe["last_month_traffic"].tolist()
# error_7 = [i for i in range(num) if temp[i] > 10000]
# select_type_dataframe["last_month_traffic"][error_7] = 0
# temp = select_type_dataframe["local_trafffic_month"].tolist()
# error_8 = [i for i in range(num) if temp[i] > 50000]
# select_type_dataframe["local_trafffic_month"][error_8] = 0
# temp = select_type_dataframe["local_caller_time"].tolist()
# error_9 = [i for i in range(num) if temp[i] > 1000]
# select_type_dataframe["local_caller_time"][error_9] = 0
# temp = select_type_dataframe["service1_caller_time"].tolist()
# error_10 = [i for i in range(num) if temp[i] > 1000]
# select_type_dataframe["service1_caller_time"][error_10] = 0
# temp = select_type_dataframe["fee_std"].tolist()
# error_11 = [i for i in range(num) if temp[i] > 500]
# select_type_dataframe["fee_std"][error_11] = 0
# # # bbox(select_type_dataframe, "fee_std")
# select_type_dataframe.to_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_sup_add_12_13_14_correct_exclude.csv")

# new_dataframe.to_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_sup_add_12_13_14_exclude.csv")

# select_type_dataframe = pd.read_csv(r"/data/projects/CCFDF_18/data/class_2_sup_add_12_13_14_exclude.csv",
#                                     encoding="utf-8", low_memory=False)
# # a = [i for i in range(len(select_type_dataframe))]
# # b = random.sample(a, 200)
# # select_type_dataframe = select_type_dataframe.iloc[b]
# dataframe_pro, pro_12, pro_13, pro_14 = pre(select_type_dataframe, para_list)
#
# dataframe_pro.to_csv(r"/data/projects/CCFDF_18/result/RF/class_2_sup_add_12_13_14_propotion.csv")
# pro_12_array = np.array(pro_12).transpose()
# pro_13_array = np.array(pro_13).transpose()
# pro_14_array = np.array(pro_14).transpose()
# pro_total = np.concatenate((pro_12_array, pro_13_array, pro_14_array), axis=1)
# dataframe_propo = pd.DataFrame(pro_total)
# dataframe_propo.to_csv(r"/data/projects/CCFDF_18/result/RF/dataframe_propo.csv")



