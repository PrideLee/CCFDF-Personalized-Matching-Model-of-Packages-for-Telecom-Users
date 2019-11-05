import pandas as pd
import numpy as np


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
        print(i)
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


####################################################  数据拆分  #####################################################
# train_raw = pd.read_csv(r'E:\CCFDF\plansmatching\data\raw data\final_data\train_all\train_all\train_all.csv',
#                         encoding="utf-8", low_memory=False)
# train_service_type = train_raw["service_type"].tolist()
# num_train = len(train_service_type)
# type_4 = [i for i in range(num_train) if train_service_type[i] == 4]
# train_4 = train_raw.iloc[type_4]
# train_4.to_csv(r'E:\CCFDF\plansmatching\data\raw data\final_data\train_all\train_all\train_4.csv')
#
#
# test_raw = pd.read_csv(r'E:\CCFDF\plansmatching\data\raw data\final_data\republish_test\republish_test\republish_test.csv',
#                         encoding="utf-8", low_memory=False)
# test_service_type = test_raw["service_type"].tolist()
# num_test = len(test_service_type)
# type_4 = [i for i in range(num_test) if test_service_type[i] == 4]
# test_4 = test_raw.iloc[type_4]
# test_4.to_csv(r'E:\CCFDF\plansmatching\data\raw data\final_data\republish_test\republish_test\test4.csv')


train_4_raw = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\train4_final.csv",
                          encoding="utf-8", low_memory=False)
test_4_raw = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\test4_final.csv",
                         encoding="utf-8", low_memory=False)
num_train = len(train_4_raw)
train_4_service_type = train_4_raw["current_service"].tolist()
train_4_service_type_encode = train_4_raw["service_type_encode"].tolist()
train_pop = train_4_raw.drop(["service_type", "service_type_encode"], axis=1)
combine_total = pd.concat([train_pop, test_4_raw])
fee_df = fee_exp(['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee'], combine_total)
fee_par = fee_par(fee_df, ['online_time', 'month_traffic', 'contract_time', 'last_month_traffic', 'local_trafffic_month',
                           'local_caller_time', 'service_caller_time_mean'])
df_feat = pay_fee(fee_par)
min_max_par = ["fee_mat_subtract_01", "fee_mat_subtract_12", "fee_mat_subtract_23", "fee_mat_subtract_0112",
               "fee_mat_subtract_1223", "fee_mat_sum_01", "fee_mat_sum_12", "fee_mat_sum_23", "fee_mat_sum_0112",
               "fee_mat_sum_0112", "pay_mean", "online_time_per_fee", "month_traffic_per_fee", "contract_time_per_fee",
               "last_month_traffic_per_fee", "local_trafffic_month_per_fee", "local_caller_time_per_fee",
               "service_caller_time_mean_per_fee", "pay_fee_subtract", "pay_total", "pay_fee_total_subtract"]

nor_df = Min_Max(min_max_par, df_feat)


train_data_pro = nor_df.iloc[0:num_train]
test_data_pro = nor_df.iloc[num_train:len(nor_df)]
train_data_pro["current_service"] = train_4_service_type
train_data_pro["service_type_encode"] = train_4_service_type_encode
train_data_pro.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\train4_feature.csv", index=False)
test_data_pro.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\test4_feature.csv", index=False)




