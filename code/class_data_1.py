import pandas as pd
import numpy as np
import random


# raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\supplement\train2\test_1.csv", encoding="utf-8",
#                        low_memory=False)
# select_1 = (raw_data["current_service"] == 7) | (raw_data["current_service"] == 8) | (
#   raw_data["current_service"] == 9)
# select_2 = (raw_data["current_service"] == 1) | (raw_data["current_service"] == 2) | (
#       raw_data["current_service"] == 3) | (raw_data["current_service"] == 4) | (
#                  raw_data["current_service"] == 5) | (raw_data["current_service"] == 6) | (
#                  raw_data["current_service"] == 11) | (raw_data["current_service"] == 12) | (
#                  raw_data["current_service"] == 13) | (raw_data["current_service"] == 14) | (
#                  raw_data["current_service"] == 0)
# class_1 = raw_data[select_1]
# class_2 = raw_data[select_2]
# class_1.to_csv(r"E:\CCFDF\plansmatching\data\raw data\supplement\train2\class_1.csv", encoding="utf-8")
# class_2.to_csv(r"E:\CCFDF\plansmatching\data\raw data\supplement\train2\class_2.csv", encoding="utf-8")


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


def month_traffic(file_dataframe):
    month_traf = file_dataframe["month_traffic"].tolist()
    month_traf_hy = []
    for i in month_traf:
        if i < 800:
            month_traf_hy.append(0)
        if 800 <= i < 2000:
            month_traf_hy.append(1)
        if 2000 <= i < 3000:
            month_traf_hy.append(2)
        if 3000 <= i < 3700:
            month_traf_hy.append(3)
        if 3700 <= i < 4800:
            month_traf_hy.append(4)
        if 4800 <= i < 5200:
            month_traf_hy.append(5)
        if 5200 <= i < 7800:
            month_traf_hy.append(6)
        if 7800 <= i < 10500:
            month_traf_hy.append(7)
        if 10500 <= i < 12000:
            month_traf_hy.append(8)
        if i >= 12000:
            month_traf_hy.append(9)
    file_dataframe["month_traffic_hierarchy_new"] = month_traf_hy
    return file_dataframe


def contract_time(file_dataframe):
    contract_ti = file_dataframe["contract_time"].tolist()
    contract_ti_hy = []
    for i in contract_ti:
        if i < 1:
            contract_ti_hy.append(0)
        if 1 <= i < 5:
            contract_ti_hy.append(1)
        if 5 <= i < 20:
            contract_ti_hy.append(2)
        if i >= 20:
            contract_ti_hy.append(3)
    file_dataframe["contract_time_hierarchy"] = contract_ti_hy
    return file_dataframe


def pay_times(file_dataframe):
    pay_ti = file_dataframe["pay_times"].tolist()
    pay_ti_hy = []
    for i in pay_ti:
        if i < 1:
            pay_ti_hy.append(0)
        if 1 <= i < 2:
            pay_ti_hy.append(1)
        if 2 <= i < 3:
            pay_ti_hy.append(3)
        if i >= 3:
            pay_ti_hy.append(4)
    file_dataframe["pay_times_hierarchy"] = pay_ti_hy
    return file_dataframe


def pay_num(file_dataframe):
    pay_n = file_dataframe["pay_num"].tolist()
    pay_n_hy = []
    for i in pay_n:
        if i > 500:
            pay_n_hy.append(600)
        else:
            pay_n_hy.append(i)
    file_dataframe["pay_nums_hierarchy"] = pay_n_hy
    return file_dataframe


def last_month_traffic(file_dataframe):
    last_month_trf = file_dataframe["last_month_traffic"]
    last_month_trf_hy = []
    for i in last_month_trf:
        if i < 1:
            last_month_trf_hy.append(0)
        if 1 <= i < 1000:
            last_month_trf_hy.append(1)
        if i >= 1000:
            last_month_trf_hy.append(2)
    file_dataframe["last_month_traffic_hierarchy"] = last_month_trf_hy
    return file_dataframe


def local_trafffic_month(file_dataframe):
    local_trafffic_mon = file_dataframe["local_trafffic_month"]
    local_trafffic_mon_hy = []
    for i in local_trafffic_mon:
        if i < 50:
            local_trafffic_mon_hy.append(0)
        if 50 <= i < 400:
            local_trafffic_mon_hy.append(1)
        if 400 <= i < 1500:
            local_trafffic_mon_hy.append(2)
        if 1500 <= i < 5000:
            local_trafffic_mon_hy.append(3)
        if 5000 <= i < 10000:
            local_trafffic_mon_hy.append(4)
        if i >= 10000:
            local_trafffic_mon_hy.append(5)
    file_dataframe["local_trafffic_month_hierarchy"] = local_trafffic_mon_hy
    return file_dataframe


def local_caller_time(file_dataframe):
    local_caller_ti = file_dataframe["local_caller_time"].tolist()
    local_caller_ti_hy = []
    for i in local_caller_ti:
        if i < 1:
            local_caller_ti_hy.append(0)
        if 1 <= i < 10:
            local_caller_ti_hy.append(1)
        if 10 <= i < 25:
            local_caller_ti_hy.append(2)
        if 25 <= i < 58:
            local_caller_ti_hy.append(3)
        if 58 <= i < 120:
            local_caller_ti_hy.append(4)
        if i >= 120:
            local_caller_ti_hy.append(5)
    file_dataframe["local_caller_time_hierarchy"] = local_caller_ti_hy
    return file_dataframe


def service1_caller_time(file_dataframe):
    service1_caller_ti = file_dataframe["service1_caller_time"].tolist()
    service1_caller_ti_hy = []
    for i in service1_caller_ti:
        if i < 1:
            service1_caller_ti_hy.append(0)
        if 1 <= i < 10:
            service1_caller_ti_hy.append(1)
        if 10 <= i < 45:
            service1_caller_ti_hy.append(2)
        if 45 <= i < 100:
            service1_caller_ti_hy.append(3)
        if i >= 100:
            service1_caller_ti_hy.append(4)
    file_dataframe["service1_caller_time_hierarchy"] = service1_caller_ti_hy
    return file_dataframe


def service2_caller_time(file_dataframe):
    service2_caller_ti = file_dataframe["service2_caller_time"].tolist()
    service2_caller_ti_hy = []
    for i in service2_caller_ti:
        if i < 2:
            service2_caller_ti_hy.append(0)
        if 2 <= i < 15:
            service2_caller_ti_hy.append(1)
        if 15 <= i < 65:
            service2_caller_ti_hy.append(2)
        if 65 <= i < 120:
            service2_caller_ti_hy.append(3)
        if 120 <= i < 190:
            service2_caller_ti_hy.append(4)
        if i >= 190:
            service2_caller_ti_hy.append(5)
    file_dataframe["service2_caller_time"] = service2_caller_ti_hy
    return file_dataframe


def service_caller_time_frc(file_dataframe):
    service1_caller_ti = file_dataframe["service1_caller_time"].tolist()
    service2_caller_ti = file_dataframe["service2_caller_time"].tolist()
    num = len(service2_caller_ti)
    service_caller_time_frc = [service2_caller_ti[i] - service1_caller_ti[i] for i in range(num)]
    file_dataframe["service_caller_time_fluctuate"] = service_caller_time_frc
    file_dataframe_norm = Min_Max(["service_caller_time_fluctuate"], file_dataframe)
    return file_dataframe_norm


def fee(par, file_dataframe):
    fee_list = []
    for i in par:
        fee_list.append(file_dataframe[i].tolist())
    fee_matrix = np.array(fee_list)
    fee_mean = np.mean(fee_matrix, axis=0).tolist()
    fee_std = np.std(fee_matrix, axis=0).tolist()
    fee_std_round = [round(i, 6) for i in fee_std]
    fee_interval_mean = []
    for j in range(len(file_dataframe)):
        fee_total = 0
        for k in range(len(fee_list) - 1):
            interval = fee_list[k + 1][j] - fee_list[k][j]
            fee_total += interval
        fee_interval_mean.append(round(fee_total / (len(fee_list) - 1), 6))
    file_dataframe["fee_mean"] = fee_mean
    file_dataframe["fee_std"] = fee_std_round
    file_dataframe["fee_fluctuate"] = fee_interval_mean
    return file_dataframe


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


def dimensionality_reduction(par, file_dataframe):
    for i in par:
        new_list = []
        par_list = file_dataframe[i].tolist()
        for j in par_list:
            if j != 0:
                new_list.append(1)
            else:
                new_list.append(0)
        dimensionlity_name = i + '_dimean'
        file_dataframe[dimensionlity_name] = new_list
    return file_dataframe


# raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\supplement\train2\class_2.csv", encoding="utf-8",
#                        low_memory=False)
# fee_data = fee(['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee'], raw_data)
# norm_par = ["online_time", "fee_mean", "fee_fluctuate", "month_traffic", "contract_time", "pay_num",
#             "last_month_traffic", "local_trafffic_month", "local_caller_time", "service1_caller_time",
#             "service2_caller_time", "age", "former_complaint_num", "former_complaint_fee"]
# norm_data = Min_Max(norm_par, fee_data)
# proprocess_data = dimensionality_reduction(
#     ["service1_caller_time", "last_month_traffic", "former_complaint_num", "former_complaint_fee"], norm_data)

# proprocess_data.to_csv(r"E:\CCFDF\plansmatching\data\raw data\supplement\train2\class_2_proprocess.csv", encoding="utf-8")


# raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_proprocess.csv", encoding="utf-8",
#                        low_memory=False)
# # raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\supplement\train2\class_2_proprocess.csv", encoding="utf-8",
# #                        low_memory=False)
# data_fee_1 = raw_data['1_total_fee'].tolist()
# data_fee_2 = raw_data['2_total_fee'].tolist()
# data_fee_3 = raw_data['3_total_fee'].tolist()
# data_fee_4 = raw_data['4_total_fee'].tolist()
# len_fee = len(data_fee_1)
# fee_1_2 = [data_fee_1[i]-data_fee_2[i] for i in range(len_fee)]
# fee_2_3 = [data_fee_2[i]-data_fee_3[i] for i in range(len_fee)]
# fee_3_4 = [data_fee_3[i]-data_fee_4[i] for i in range(len_fee)]
# fee_12_23 = [fee_1_2[i]-fee_2_3[i] for i in range(len_fee)]
# fee_23_34 = [fee_2_3[i]-fee_3_4[i] for i in range(len_fee)]
# mean_2 = [(fee_12_23[i]+fee_23_34[i])/2 for i in range(len_fee)]
# raw_data["fee_mean_2"] = mean_2
# raw_data = Min_Max(["fee_mean_2"], raw_data)
# raw_data.to_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_proprocess_2.csv", encoding="utf-8")

# raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_sup.csv", encoding="utf-8", low_memory=False)
# f1 = month_traffic(raw_data)
# f2 = contract_time(f1)
# f3 = pay_times(f2)
# f4 = pay_num(f3)
# f5 = last_month_traffic(f4)
# f6 = local_trafffic_month(f5)
# f7 = local_caller_time(f6)
# f8 = service1_caller_time(f7)
# f9 = service2_caller_time(f8)
# f10 = service_caller_time_frc(f9)
# f10.to_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_sup_add_0.csv", encoding="utf-8")


# test = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\supplement\train2\test.csv", encoding="utf-8",
#                        low_memory=False)
# label = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\supplement\train2\standard.csv", encoding="utf-8",
#                        low_memory=False)
#
# user_id_1 = test["user_id"].tolist()
# user_id_2 = label["user_id"].tolist()
# type = label["current_service"].tolist()
# num = len(user_id_1)
# site_label = [user_id_2.index(user_id_1[i]) for i in range(num)]
# type_service = [type[i] for i in site_label]
# test["service_current"] = type_service
# test.to_csv(r"E:\CCFDF\plansmatching\data\raw data\supplement\train2\test_1.csv", encoding="utf-8")

# dataframe_balance = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_sup_add_0_balance.csv", encoding="utf-8",
#                                 low_memory=False)
# num_total = len(dataframe_balance)
# index_to = range(num_total)
# sample_site = random.sample(index_to, round(num_total * 0.1))
# sample_dataframe = dataframe_balance.iloc[sample_site]
# service_type = sample_dataframe["current_service"].tolist()
# service_type_new = []
# for i in service_type:
#     if i == 11:
#         service_type_new.append(7)
#     if i == 12:
#         service_type_new.append(8)
#     if i == 13:
#         service_type_new.append(9)
#     if i == 14:
#         service_type_new.append(10)
#     if i in [0, 1, 2, 3, 4, 5, 6]:
#         service_type_new.append(i)
# sample_dataframe["current_sample_reencode"] = service_type_new
# sample_dataframe.to_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_sup_add_0_balance_sample.csv")
#

raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_sup_add_0.csv", encoding="utf-8",
                       low_memory=False)
norm_par = ["online_time", "fee_mean", "fee_fluctuate", "month_traffic", "contract_time", "pay_num",
            "last_month_traffic", "local_trafffic_month", "local_caller_time", "service1_caller_time",
            "service2_caller_time", "age", "former_complaint_num", "former_complaint_fee", 'fee_mean_2',
            'service_caller_time_fluctuate']
norm_data = Min_Max(norm_par, raw_data)
norm_data.to_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_sup_add_0_correct.csv")
