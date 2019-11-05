import pandas as pd
import numpy as np


def min_fee(dataframe, para_fee):
    fee_array = np.array(dataframe[para_fee])
    fee_min = np.min(fee_array, axis=1)
    dataframe["fee_min"] = fee_min
    return dataframe


def traffic_current_month(dataframe):
    month_traffic = dataframe["month_traffic"].tolist()
    last_month_traffic = dataframe["last_month_traffic"].tolist()
    num = len(month_traffic)
    traffic_current_month = [month_traffic[i] - last_month_traffic[i] for i in range(num)]
    dataframe["traffic_current_month"] = traffic_current_month
    return dataframe

def fee_exp(par_fee, dataframe):
    fee_mat = np.array(dataframe[par_fee])
    fee_mat_subtract_01 = fee_mat[:, 0] - fee_mat[:, 1]
    fee_mat_subtract_12 = fee_mat[:, 1] - fee_mat[:, 2]
    fee_mat_subtract_23 = fee_mat[:, 2] - fee_mat[:, 3]
    fee_mat_subtract_0112 = fee_mat_subtract_01 - fee_mat_subtract_12
    fee_mat_subtract_1223 = fee_mat_subtract_12 - fee_mat_subtract_23
    fee_mat_sum_01 = fee_mat[:, 0] + fee_mat[:, 1]
    fee_mat_sum_12 = fee_mat[:, 1] + fee_mat[:, 2]
    fee_mat_sum_23 = fee_mat[:, 2] + fee_mat[:, 3]
    fee_mat_sum_0112 = fee_mat_sum_01 + fee_mat_sum_12
    fee_mat_sum_1223 = fee_mat_sum_12 + fee_mat_sum_23
    dataframe["fee_mat_subtract_01"] = fee_mat_subtract_01
    dataframe["fee_mat_subtract_12"] = fee_mat_subtract_12
    dataframe["fee_mat_subtract_23"] = fee_mat_subtract_23
    dataframe["fee_mat_subtract_0112"] = fee_mat_subtract_0112
    dataframe["fee_mat_subtract_1223"] = fee_mat_subtract_1223
    dataframe["fee_mat_sum_01"] = fee_mat_sum_01
    dataframe["fee_mat_sum_12"] = fee_mat_sum_12
    dataframe["fee_mat_sum_23"] = fee_mat_sum_23
    dataframe["fee_mat_sum_0112"] = fee_mat_sum_0112
    dataframe["fee_mat_sum_1223"] = fee_mat_sum_1223
    return dataframe


def fee_par(dataframe, par_list):
    fee_mean = dataframe['fee_mean'].tolist()
    num_fee = len(fee_mean)
    for i in par_list:
        temp = dataframe[i].tolist()
        temp_fee_per = []
        for j in range(num_fee):
            if temp[j] == 0:
                temp_fee_per.append(99999)
            else:
                temp_fee_per.append(round(fee_mean[j]/temp[j], 6))
        dataframe[i + '_per_fee'] = temp_fee_per
    return dataframe


def pay_fee(dataframe):
    pay = dataframe["pay_num"].tolist()
    fee = dataframe["fee_mean"].tolist()
    pay_times = dataframe["pay_times"].tolist()
    num = len(pay)
    pay_fee_subtract = [pay[i]-fee[i] for i in range(num)]
    pay_total = [round(pay[i] * pay_times[i], 6) for i in range(num)]
    pay_fee_total_subtract = [round((fee[i] * 4 - pay_total[i]), 6) for i in range(num)]
    dataframe["pay_fee_subtract"] = pay_fee_subtract
    dataframe["pay_total"] = pay_total
    dataframe["pay_fee_total_subtract"] = pay_fee_total_subtract
    return dataframe



def fee(par, file_dataframe):
    fee_list = []
    for i in par:
        fee_list.append(file_dataframe[i].tolist())
    fee_matrix = np.array(fee_list)
    fee_max = fee_matrix.max(axis=0)
    fee_min = fee_matrix.min(axis=0)
    fee_dis = (fee_max - fee_min).tolist()
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
    file_dataframe["fee_distance"] = fee_dis
    return file_dataframe


def fee_2(raw_data):
    data_fee_1 = raw_data['1_total_fee'].tolist()
    data_fee_2 = raw_data['2_total_fee'].tolist()
    data_fee_3 = raw_data['3_total_fee'].tolist()
    data_fee_4 = raw_data['4_total_fee'].tolist()
    len_fee = len(data_fee_1)
    fee_1_2 = [data_fee_1[i] - data_fee_2[i] for i in range(len_fee)]
    fee_2_3 = [data_fee_2[i] - data_fee_3[i] for i in range(len_fee)]
    fee_3_4 = [data_fee_3[i] - data_fee_4[i] for i in range(len_fee)]
    fee_12_23 = [fee_1_2[i] - fee_2_3[i] for i in range(len_fee)]
    fee_23_34 = [fee_2_3[i] - fee_3_4[i] for i in range(len_fee)]
    mean_2 = [(fee_12_23[i] + fee_23_34[i]) / 2 for i in range(len_fee)]
    raw_data["fee_mean_2"] = mean_2
    return raw_data


def pay_mean(file_dataframe):
    pay_time = file_dataframe["pay_times"].tolist()
    pay_fee = file_dataframe["pay_num"].tolist()
    num_total = len(pay_fee)
    mean_pay = []
    for i in range(num_total):
        if pay_time[i] == 0:
            mean_pay.append(pay_fee[i])
        else:
            mean_pay.append(round(pay_fee[i]/pay_time[i], 6))
    file_dataframe["pay_mean"] = mean_pay
    return file_dataframe



def service_caller_time_frc(file_dataframe):
    service1_caller_ti = file_dataframe["service1_caller_time"].tolist()
    service2_caller_ti = file_dataframe["service2_caller_time"].tolist()
    num = len(service2_caller_ti)
    service_caller_time_frc = [service2_caller_ti[i] - service1_caller_ti[i] for i in range(num)]
    service_caller_time_mean = [round((service2_caller_ti[i] + service1_caller_ti[i])/2, 6) for i in range(num)]
    file_dataframe["service_caller_time_fluctuate"] = service_caller_time_frc
    file_dataframe["service_caller_time_mean"] = service_caller_time_mean
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



train_data_fr = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\class_2.csv",
                       encoding="utf-8", low_memory=False)
num_train = len(train_data_fr)
train_pop_id = train_data_fr.drop(["current_service", "service_type_encode"], axis=1)
test_data_fr = pd.read_csv(
    r"E:\CCFDF\plansmatching\data\raw data\final_data\republish_test\republish_test\republish_test_type4_new.csv",
    encoding="utf-8", low_memory=False)
combine_data = pd.concat([train_data_fr, test_data_fr])


fee_data = fee(['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee'], combine_data)
fee_2_data = fee_2(fee_data)
time_frc = service_caller_time_frc(fee_2_data)
pay = pay_mean(time_frc)
para_list = ["online_time", "fee_mean", "fee_std", "fee_fluctuate", "fee_distance", "month_traffic", "contract_time",
             "pay_num", "pay_mean",
             "last_month_traffic", "local_trafffic_month", "local_caller_time", "service1_caller_time",
             "service2_caller_time", "age", "former_complaint_num", "former_complaint_fee", 'fee_mean_2',
             "service_caller_time_fluctuate", "service_caller_time_mean",
             "1_total_fee", "2_total_fee", "3_total_fee", "4_total_fee"]
final_data = Min_Max(para_list, pay)
train_data_pro = final_data.iloc[0:num_train]

test_data_pro = final_data.iloc[num_train:len(final_data)]

train_data_pro.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\train4_final.csv")
test_data_pro.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\test4_final.csv")

