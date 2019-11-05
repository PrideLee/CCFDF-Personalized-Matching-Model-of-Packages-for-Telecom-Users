import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\train_all\train_all.csv", encoding="utf-8",
#                        low_memory=False)
#
# service_type = raw_data["current_service"].tolist()
# new_type = []
# for i in service_type:
#     if i == 89950166:
#         new_type.append(0)
#     if i == 89950167:
#         new_type.append(1)
#     if i == 89950168:
#         new_type.append(2)
#     if i == 99999825:
#         new_type.append(3)
#     if i == 99999826:
#         new_type.append(4)
#     if i == 99999827:
#         new_type.append(5)
#     if i == 99999828:
#         new_type.append(6)
#     if i == 99999830:
#         new_type.append(7)
#     if i == 90063345:
#         new_type.append(8)
#     if i == 90109916:
#         new_type.append(9)
#     if i == 90155946:
#         new_type.append(10)
# raw_data["service_type_encode"] = new_type
# num_total = len(new_type)
# class_1 = [8, 9, 10]
# class_2 = [0, 1, 2, 3, 4, 5, 6, 7]
# class_1_site = []
# class_2_site = []
# for i in range(num_total):
#     if new_type[i] in class_1:
#         class_1_site.append(i)
#     if new_type[i] in class_2:
#         class_2_site.append(i)
# class_2_data = raw_data.iloc[class_2_site]
# class_1_data = raw_data.iloc[class_1_site]
# class_1_data.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\class_1.csv")
# class_2_data.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\class_2.csv")


def class_sample(raw_data):
    service_type = raw_data["current_service"].tolist()
    new_type = []
    for i in service_type:
        if i == 89950166:
            new_type.append(0)
        if i == 89950167:
            new_type.append(1)
        if i == 89950168:
            new_type.append(2)
        if i == 99999825:
            new_type.append(3)
        if i == 99999826:
            new_type.append(4)
        if i == 99999827:
            new_type.append(5)
        if i == 99999828:
            new_type.append(6)
        if i == 99999830:
            new_type.append(7)
        if i == 90063345:
            new_type.append(8)
        if i == 90109916:
            new_type.append(9)
        if i == 90155946:
            new_type.append(10)
    raw_data["service_type_encode"] = new_type
    num_total = len(new_type)
    class_1 = [8, 9, 10]
    class_2 = [0, 1, 2, 3, 4, 5, 6, 7]
    class_1_site = []
    class_2_site = []
    for i in range(num_total):
        if new_type[i] in class_1:
            class_1_site.append(i)
        if new_type[i] in class_2:
            class_2_site.append(i)
    class_2_data = raw_data.iloc[class_2_site]
    class_1_data = raw_data.iloc[class_1_site]
    return class_1_data, class_2_data


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
    file_dataframe["service_caller_time_fluctuate"] = service_caller_time_frc
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


def bbox(raw_data, att):
    y = raw_data[att].tolist()
    labels = [att]
    plt.boxplot(y, labels=labels, sym='o')
    plt.grid(True)
    plt.show()


def error_data(select_type_dataframe):
    num = len(select_type_dataframe)
    temp = select_type_dataframe["local_trafffic_month"].tolist()
    error_1 = [i for i in range(num) if temp[i] > 10 ** 7]
    select_type_dataframe["local_trafffic_month"][error_1] = 0
    temp = select_type_dataframe["pay_num"].tolist()
    error_2 = [i for i in range(num) if temp[i] > 30000]
    select_type_dataframe["pay_num"][error_2] = 0
    temp = select_type_dataframe["fee_mean"].tolist()
    error_3 = [i for i in range(num) if temp[i] > 15000]
    select_type_dataframe["fee_mean"][error_3] = 0
    temp = select_type_dataframe["fee_mean_2"].tolist()
    error_4 = [i for i in range(num) if (temp[i] > 10000) | (temp[i] < -10000)]
    select_type_dataframe["fee_mean_2"][error_4] = 0
    temp = select_type_dataframe["month_traffic"].tolist()
    error_5 = [i for i in range(num) if temp[i] > 10000]
    select_type_dataframe["month_traffic"][error_5] = 0
    temp = select_type_dataframe["pay_times"].tolist()
    error_6 = [i for i in range(num) if temp[i] > 10]
    select_type_dataframe["pay_times"][error_6] = 0
    temp = select_type_dataframe["last_month_traffic"].tolist()
    error_7 = [i for i in range(num) if temp[i] > 10000]
    select_type_dataframe["last_month_traffic"][error_7] = 0
    temp = select_type_dataframe["local_trafffic_month"].tolist()
    error_8 = [i for i in range(num) if temp[i] > 50000]
    select_type_dataframe["local_trafffic_month"][error_8] = 0
    temp = select_type_dataframe["local_caller_time"].tolist()
    error_9 = [i for i in range(num) if temp[i] > 1000]
    select_type_dataframe["local_caller_time"][error_9] = 0
    temp = select_type_dataframe["service1_caller_time"].tolist()
    error_10 = [i for i in range(num) if temp[i] > 1000]
    select_type_dataframe["service1_caller_time"][error_10] = 0
    temp = select_type_dataframe["fee_std"].tolist()
    error_11 = [i for i in range(num) if temp[i] > 500]
    select_type_dataframe["fee_std"][error_11] = 0
    return select_type_dataframe


def error_data_class_1(select_type_dataframe):
    num = len(select_type_dataframe)
    temp = select_type_dataframe["local_trafffic_month"].tolist()
    error_1 = [i for i in range(num) if temp[i] > 50000]
    select_type_dataframe["local_trafffic_month"][error_1] = 0
    temp = select_type_dataframe["pay_num"].tolist()
    error_2 = [i for i in range(num) if temp[i] > 300]
    select_type_dataframe["pay_num"][error_2] = 0
    temp = select_type_dataframe["fee_mean"].tolist()
    error_3 = [i for i in range(num) if temp[i] > 150]
    select_type_dataframe["fee_mean"][error_3] = 0
    temp = select_type_dataframe["fee_mean_2"].tolist()
    error_4 = [i for i in range(num) if (temp[i] > 100) | (temp[i] < -100)]
    select_type_dataframe["fee_mean_2"][error_4] = 0
    temp = select_type_dataframe["month_traffic"].tolist()
    error_5 = [i for i in range(num) if temp[i] > 10000]
    select_type_dataframe["month_traffic"][error_5] = 0
    temp = select_type_dataframe["pay_times"].tolist()
    error_6 = [i for i in range(num) if temp[i] > 10]
    select_type_dataframe["pay_times"][error_6] = 0
    temp = select_type_dataframe["last_month_traffic"].tolist()
    error_7 = [i for i in range(num) if temp[i] > 500]
    select_type_dataframe["last_month_traffic"][error_7] = 0
    temp = select_type_dataframe["local_trafffic_month"].tolist()
    error_8 = [i for i in range(num) if temp[i] > 50000]
    select_type_dataframe["local_trafffic_month"][error_8] = 0
    temp = select_type_dataframe["local_caller_time"].tolist()
    error_9 = [i for i in range(num) if temp[i] > 8000]
    select_type_dataframe["local_caller_time"][error_9] = 0
    temp = select_type_dataframe["service1_caller_time"].tolist()
    error_10 = [i for i in range(num) if temp[i] > 700]
    select_type_dataframe["service1_caller_time"][error_10] = 0
    temp = select_type_dataframe["fee_std"].tolist()
    error_11 = [i for i in range(num) if temp[i] > 70]
    select_type_dataframe["fee_std"][error_11] = 0
    temp = select_type_dataframe["pay_mean"].tolist()
    error_12 = [i for i in range(num) if temp[i] > 300]
    select_type_dataframe["pay_mean"][error_12] = 0
    return select_type_dataframe



# raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\class_2.csv", encoding="utf-8",
#                        low_memory=False)
# fee_data = fee(['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee'], raw_data)
# fee_2_data = fee_2(fee_data)
# time_frc = service_caller_time_frc(fee_2_data)
# correct_data = error_data(time_frc)
# para_list = ["online_time", "fee_mean", "fee_std", "fee_fluctuate", "month_traffic", "contract_time", "pay_num",
#              "last_month_traffic", "local_trafffic_month", "local_caller_time", "service1_caller_time",
#              "service2_caller_time", "age", "former_complaint_num", "former_complaint_fee", 'fee_mean_2',
#              "service_caller_time_fluctuate", "1_total_fee", "2_total_fee", "3_total_fee", "4_total_fee"]
# final_data = Min_Max(para_list, correct_data)
# final_data.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\class_2_pro_new.csv")


# raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\class_2_pro.csv", encoding="utf-8",
#                        low_memory=False)
# norm_list = ["1_total_fee", "2_total_fee", "3_total_fee", "4_total_fee"]
# final_data = Min_Max(norm_list, raw_data)
# final_data.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\class_2_pro_1.csv")


##################     test setting     #############################


# raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\republish_test\republish_test_1.csv", encoding="utf-8",
#                        low_memory=False)
#
# fee_data = fee(['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee'], raw_data)
# fee_2_data = fee_2(fee_data)
# time_frc = service_caller_time_frc(fee_2_data)
# correct_data = errr_data(time_frc)
# para_list = ["online_time", "fee_mean", "fee_std", "fee_fluctuate", "month_traffic", "contract_time", "pay_num",
#              "last_month_traffic", "local_trafffic_month", "local_caller_time", "service1_caller_time",
#              "service2_caller_time", "age", "former_complaint_num", "former_complaint_fee", 'fee_mean_2',
#              "service_caller_time_fluctuate", '1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee']
# final_data = Min_Max(para_list, time_frc)
# final_data.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\republish_test\republish_test\republish_test_pro.csv")
#


# raw_data = pd.read_csv(
#     r"E:\CCFDF\plansmatching\data\raw data\final_data\republish_test\republish_test\repiulish_test_duplicate.csv",
#     encoding="utf-8", low_memory=False)
# type_service = raw_data["service_type"].tolist()
# num = len(type_service)
# service_type_4 = [i for i in range(num) if type_service[i] == 4]
# dataframe_4 = raw_data.iloc[service_type_4]
# dataframe_4.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\republish_test\republish_test\republish_test_type4_new.csv")
#


# raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\republish_test\republish_test\republish_test_type4_new.csv", encoding="utf-8",
#                        low_memory=False)
#
# fee_data = fee(['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee'], raw_data)
# fee_2_data = fee_2(fee_data)
# time_frc = service_caller_time_frc(fee_2_data)
# correct_data = error_data(time_frc)
# para_list = ["online_time", "fee_mean", "fee_std", "fee_fluctuate", "month_traffic", "contract_time", "pay_num",
#              "last_month_traffic", "local_trafffic_month", "local_caller_time", "service1_caller_time",
#              "service2_caller_time", "age", "former_complaint_num", "former_complaint_fee", 'fee_mean_2',
#              "service_caller_time_fluctuate", '1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee']
# final_data = Min_Max(para_list, time_frc)
# final_data.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\republish_test\republish_test\republish_test_type4_new_pro.csv")
#


# raw_data = pd.read_csv(
#     r"E:\CCFDF\plansmatching\data\raw data\final_data\republish_test\republish_test\republish_test_pro.csv",
#     encoding="utf-8", low_memory=False)
# type_service = raw_data["service_type"].tolist()
# num = len(type_service)
# service_type_4 = [i for i in range(num) if type_service[i] == 4]
# dataframe_4 = raw_data.iloc[service_type_4]
# dataframe_4.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\republish_test\republish_test\republish_test_type4.csv")
# #

# num = len(data_raw)

# temp = data_raw["4_total_fee"].tolist()
# temp_1 = []
# for i in temp:
#     if i == '\\N':
#         temp_1.append(0)
#     else:
#         temp_1.append(float(i))
# error_site = [i for i in range(num) if temp_1[i] > 2000]
# print(error_site)
#
# bbox(data_raw, "3_total_fee")

# test_data = pd.read_csv(
#     r"E:\CCFDF\plansmatching\data\raw data\final_data\republish_test\republish_test\republish_test.csv",
#     encoding="utf-8", low_memory=False)
# data_su = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\my_submit.csv", encoding="utf-8",
#                       low_memory=False)
# id_user_test = test_data["user_id"].tolist()
# id_user_submit = data_su["user_id"].tolist()
# result_su = data_su["current_service"].tolist()
# type = test_data["service_type"].tolist()
# num_total = len(type)
# type_1 = [i for i in range(num_total) if type[i] == 1]
# type_1_id = [id_user_test[j] for j in type_1]
# type_3 = [i for i in range(num_total) if type[i] == 3]
# type_3_id = [id_user_test[j] for j in type_3]
# result_1 = [result_su[id_user_submit.index(i)] for i in type_1_id]
# result_3 = [result_su[id_user_submit.index(i)] for i in type_3_id]
# id_remain = type_1_id + type_3_id
# result_remain = result_1 + result_3
# remain_result = pd.DataFrame({"user_id": id_remain, "current_service": result_remain})
# remain_result.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\result_part_1.csv")
#

# part_0 = pd.read_csv(
#     r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\result_part_0.csv",
#     encoding="utf-8", low_memory=False)
# part_1 = pd.read_csv(
#     r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\result_part_1.csv",
#     encoding="utf-8", low_memory=False)
# user_id = part_0["user_id"].tolist() + part_1["user_id"].tolist()
# current_service = part_0["current_service"].tolist() + part_1["current_service"].tolist()
# final_result = pd.DataFrame({"user_id": user_id, "current_service": current_service})
# final_result.to_csv( r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\result_0.csv")
#

# part_0 = pd.read_csv(
#     r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\result_0.csv",
#     encoding="utf-8", low_memory=False)
# raw_data = pd.read_csv(
#     r"E:\CCFDF\plansmatching\data\raw data\final_data\republish_test\republish_test\republish_test.csv",
#     encoding="utf-8", low_memory=False)
#
# id_raw = raw_data["user_id"].tolist()
# id_result = part_0["user_id"].tolist()
# current_service = part_0["current_service"].tolist()
# site = [id_result.index(i) for i in id_raw]
# result_final = [current_service[i] for i in site]
# final_result = pd.DataFrame({"user_id": id_raw, "current_service": result_final})
# final_result.to_csv( r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\result_0_new.csv")


# raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\combine_4.csv", encoding="utf-8", low_memory=False)
# fee_data = fee(['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee'], raw_data)
# fee_2_data = fee_2(fee_data)
# time_frc = service_caller_time_frc(fee_2_data)
# correct_data = error_data(time_frc)
# para_list = ["online_time", "fee_mean", "fee_std", "fee_fluctuate", "month_traffic", "contract_time", "pay_num",
#              "last_month_traffic", "local_trafffic_month", "local_caller_time", "service1_caller_time",
#              "service2_caller_time", "age", "former_complaint_num", "former_complaint_fee", 'fee_mean_2',
#              "service_caller_time_fluctuate", '1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee']
# final_data = Min_Max(para_list, time_frc)
# final_data.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\combine_4_norm.csv")




# data_raw = pd.read_csv(
#     r"E:\CCFDF\plansmatching\data\raw data\final_data\republish_test\republish_test_duplicate_1.csv",
#     encoding="utf-8", low_memory=False)
# train_1 = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\class_1.csv", encoding="utf-8",
#                       low_memory=False)
# type_ser = data_raw["service_type"].tolist()
# num_tol = len(type_ser)
# test_1 = [i for i in range(num_tol) if type_ser[i] == 1]
# class_1 = data_raw.iloc[test_1]
# class_1_total = pd.concat([train_1, class_1])
# fee_data = fee(['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee'], class_1_total)
# fee_2_data = fee_2(fee_data)
# time_frc = service_caller_time_frc(fee_2_data)
# pay_means = pay_mean(time_frc)
# # class_1_total.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\republish_test\republish_test\class_1_total.csv")
#
#
# class_1_total = pd.read_csv(
#     r"E:\CCFDF\plansmatching\data\raw data\final_data\republish_test\republish_test\class_1_total.csv",
#     encoding="utf-8", low_memory=False)
# # bbox(class_1_total, 'pay_mean')

# correct_data = error_data_class_1(pay_means)
# para_list = ["online_time", "fee_mean", "fee_std", "fee_fluctuate", "month_traffic", "contract_time", "pay_num", "pay_mean",
#              "last_month_traffic", "local_trafffic_month", "local_caller_time", "service1_caller_time",
#              "service2_caller_time", "age", "former_complaint_num", "former_complaint_fee", 'fee_mean_2',
#              "service_caller_time_fluctuate", '1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee']
# final_data = Min_Max(para_list, correct_data)
# final_data.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\class_1_total_new.csv")
