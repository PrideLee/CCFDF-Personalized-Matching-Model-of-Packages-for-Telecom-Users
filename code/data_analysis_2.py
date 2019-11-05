import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


def scater_service_type(att):
    type_0 = raw_data[raw_data["current_service"] == 89016252][att].tolist()
    type_1 = raw_data[raw_data["current_service"] == 89016253][att].tolist()
    type_2 = raw_data[raw_data["current_service"] == 89016259][att].tolist()
    type_3 = raw_data[raw_data["current_service"] == 89950166][att].tolist()
    type_4 = raw_data[raw_data["current_service"] == 89950167][att].tolist()
    type_5 = raw_data[raw_data["current_service"] == 89950168][att].tolist()
    type_6 = raw_data[raw_data["current_service"] == 99999825][att].tolist()
    type_7 = raw_data[raw_data["current_service"] == 99999826][att].tolist()
    type_8 = raw_data[raw_data["current_service"] == 99999827][att].tolist()
    type_9 = raw_data[raw_data["current_service"] == 99999828][att].tolist()
    type_10 = raw_data[raw_data["current_service"] == 99999830][att].tolist()
    type_11 = raw_data[raw_data["current_service"] == 90063345][att].tolist()
    type_12 = raw_data[raw_data["current_service"] == 90109916][att].tolist()
    type_13 = raw_data[raw_data["current_service"] == 90155946][att].tolist()
    type_14 = raw_data[raw_data["current_service"] == 99104722][att].tolist()

    x_1 = [1 + random.random() for i in range(len(type_0))]
    x_2 = [3 + random.random() for i in range(len(type_1))]
    x_3 = [5 + random.random() for i in range(len(type_2))]
    x_4 = [7 + random.random() for i in range(len(type_3))]
    x_5 = [9 + random.random() for i in range(len(type_4))]
    x_6 = [11 + random.random() for i in range(len(type_5))]
    x_7 = [13 + random.random() for i in range(len(type_6))]
    x_8 = [15 + random.random() for i in range(len(type_7))]
    x_9 = [17 + random.random() for i in range(len(type_8))]
    x_10 = [19 + random.random() for i in range(len(type_9))]
    x_11 = [21 + random.random() for i in range(len(type_10))]
    x_12 = [23 + random.random() for i in range(len(type_11))]
    x_13 = [25 + random.random() for i in range(len(type_12))]
    x_14 = [27 + random.random() for i in range(len(type_13))]
    x_15 = [29 + random.random() for i in range(len(type_14))]
    plt.scatter(x_1, type_0, c="red", s=0.03)
    plt.scatter(x_2, type_1, c="red", s=0.03)
    plt.scatter(x_3, type_2, c="red", s=0.03)
    plt.scatter(x_4, type_3, c="red", s=0.03)
    plt.scatter(x_5, type_4, c="red", s=0.03)
    plt.scatter(x_6, type_5, c="red", s=0.03)
    plt.scatter(x_7, type_6, c="red", s=0.03)
    plt.scatter(x_8, type_7, c="red", s=0.03)
    plt.scatter(x_9, type_8, c="red", s=0.03)
    plt.scatter(x_10, type_9, c="red", s=0.03)
    plt.scatter(x_11, type_10, c="red", s=0.03)
    plt.scatter(x_12, type_11, c="red", s=0.03)
    plt.scatter(x_13, type_12, c="red", s=0.03)
    plt.scatter(x_14, type_13, c="red", s=0.03)
    plt.scatter(x_15, type_14, c="red", s=0.03)
    plt.xlabel('service_type')
    plt.ylabel(att)
    plt.grid(True)
    # plt.title("service_type-1_total_fee scatter")
    plt.show()


def bbox(att):
    type_0 = raw_data[raw_data["current_service"] == 89016252][att].tolist()
    type_1 = raw_data[raw_data["current_service"] == 89016253][att].tolist()
    type_2 = raw_data[raw_data["current_service"] == 89016259][att].tolist()
    type_3 = raw_data[raw_data["current_service"] == 89950166][att].tolist()
    type_4 = raw_data[raw_data["current_service"] == 89950167][att].tolist()
    type_5 = raw_data[raw_data["current_service"] == 89950168][att].tolist()
    type_6 = raw_data[raw_data["current_service"] == 99999825][att].tolist()
    type_7 = raw_data[raw_data["current_service"] == 99999826][att].tolist()
    type_8 = raw_data[raw_data["current_service"] == 99999827][att].tolist()
    type_9 = raw_data[raw_data["current_service"] == 99999828][att].tolist()
    type_10 = raw_data[raw_data["current_service"] == 99999830][att].tolist()
    type_11 = raw_data[raw_data["current_service"] == 90063345][att].tolist()
    type_12 = raw_data[raw_data["current_service"] == 90109916][att].tolist()
    type_13 = raw_data[raw_data["current_service"] == 90155946][att].tolist()
    type_14 = raw_data[raw_data["current_service"] == 99104722][att].tolist()
    y = np.transpose(np.array([type_0, type_1, type_2, type_3, type_4, type_5, type_6, type_7, type_8, type_9, type_10, type_11, type_12, type_13, type_14]))
    # y = np.transpose(np.array(
    #     [type_0, type_1, type_2, type_3, type_4, type_5, type_6, type_7, type_8, type_9, type_10, type_11, type_12,
    #      type_13]))
    labels = ["89016252", "89016253", "89016259", "89950166", "89950167", "89950168", "99999825", "99999826", "99999827", "99999828", "99999830", "90063345", "90109916", "90155946", "99104722"]
    # labels = ["89016252", "89016253", "89016259", "89950166", "89950167", "89950168", "99999825", "99999826",
    #           "99999827", "99999828", "99999830", "90063345", "90109916", "90155946"]
    plt.boxplot(y, labels=labels, sym='o')
    plt.grid(True)
    plt.show()


def binary_value_distribution(att):
    type_0 = raw_data[raw_data["current_service"] == 89016252][att]
    type_1 = raw_data[raw_data["current_service"] == 89016253][att]
    type_2 = raw_data[raw_data["current_service"] == 89016259][att]
    type_3 = raw_data[raw_data["current_service"] == 89950166][att]
    type_4 = raw_data[raw_data["current_service"] == 89950167][att]
    type_5 = raw_data[raw_data["current_service"] == 89950168][att]
    type_6 = raw_data[raw_data["current_service"] == 99999825][att]
    type_7 = raw_data[raw_data["current_service"] == 99999826][att]
    type_8 = raw_data[raw_data["current_service"] == 99999827][att]
    type_9 = raw_data[raw_data["current_service"] == 99999828][att]
    type_10 = raw_data[raw_data["current_service"] == 99999830][att]
    type_11 = raw_data[raw_data["current_service"] == 90063345][att]
    type_12 = raw_data[raw_data["current_service"] == 90109916][att]
    type_13 = raw_data[raw_data["current_service"] == 90155946][att]
    type_14 = raw_data[raw_data["current_service"] == 99104722][att]
    print(type_0.value_counts())
    print(type_1.value_counts())
    print(type_2.value_counts())
    print(type_3.value_counts())
    print(type_4.value_counts())
    print(type_5.value_counts())
    print(type_6.value_counts())
    print(type_7.value_counts())
    print(type_8.value_counts())
    print(type_9.value_counts())
    print(type_10.value_counts())
    print(type_11.value_counts())
    print(type_12.value_counts())
    print(type_13.value_counts())
    print(type_14.value_counts())


def dimensionality_reduction(par):
    new_list = []
    par_list = raw_data[par].tolist()
    for i in par_list:
        if i != 0:
            new_list.append(1)
        else:
            new_list.append(0)
    return new_list


def parallel_coordinates(dataframe, paralist):
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
    list_14_sample_num = round(len(list_14) * 0.68)
    list_14_sample = random.sample(list_14, list_14_sample_num)
    dataframe_12 = dataframe.iloc[list_12]
    dataframe_13 = dataframe.iloc[list_13]
    dataframe_14 = dataframe.iloc[list_14_sample]
    par_num = len(paralist)
    x_type = [i*3 for i in range(par_num)]
    x_12, y_12 = parallel_coordinates_part(dataframe_12, x_type, paralist)
    num_12 = len(dataframe_12)
    print(1)
    for i in range(num_12):
        plt.plot(x_12[i], y_12[i], 'r')
        plt.hold
    x_13, y_13 = parallel_coordinates_part(dataframe_13, x_type, paralist)
    num_13 = len(dataframe_13)
    print(2)
    for i in range(num_13):
        plt.plot(x_13[i], y_13[i], 'g')
        plt.hold
    x_14, y_14 = parallel_coordinates_part(dataframe_14, x_type, paralist)
    num_14 = len(dataframe_14)
    print(3)
    for i in range(num_14):
        plt.plot(x_14[i], y_14[i], 'b')
        plt.hold
    plt.show()


def parallel_coordinates_part(dataframe, x, parlist):
    num = len(dataframe)
    att = [[m + random.random() for m in x] for n in range(num)]
    y = [[dataframe.iloc[n][m] for m in parlist] for n in range(num)]
    return att, y


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
    bound_12 = max(round(num_12 * 0.001), 100)
    para_12_bound = [paralist_12[j][num_12] - paralist_12[j][0] for j in range(para_num)]
    pre_12 = [[] for i in range(para_num)]
    num_13 = len(paralist_13[0])
    bound_13 = max(round(num_13 * 0.001), 100)
    para_13_bound = [paralist_13[j][num_13] - paralist_13[j][0] for j in range(para_num)]
    pre_13 = [[] for i in range(para_num)]
    num_14 = len(paralist_14[0])
    bound_14 = max(round(num_14 * 0.001), 100)
    para_14_bound = [paralist_14[j][num_14] - paralist_14[j][0] for j in range(para_num)]
    pre_14 = [[] for i in range(para_num)]
    for i in range(num):
        for j in range(para_num):
            temp = dataframe.iloc[i][paralist[j]]
            for k in range(num_12):
                if temp < paralist_12[j][k]:
                    paralist_12[j].insert(k, temp)
                    break
            up = min(k + bound_12, num_12)
            down = max(k - bound_12, 0)
            pre_12[j].append(round((paralist_12[j][up] - paralist_12[j][down])/para_12_bound, 6))
            for k in range(num_13):
                if temp < paralist_13[j][k]:
                    paralist_13[j].insert(k, temp)
                    break
            up = min(k + bound_13, num_13)
            down = max(k - bound_13, 0)
            pre_13[j].append(round((paralist_13[j][up] - paralist_13[j][down])/para_13_bound, 6))
            for k in range(num_14):
                if temp < paralist_14[j][k]:
                    paralist_14[j].insert(k, temp)
                    break
            up = min(k + bound_14, num_14)
            down = max(k - bound_14, 0)
            pre_14[j].append(round((paralist_14[j][up] - paralist_14[j][down])/para_14_bound, 6))
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
        dataframe[paralist[j] + "probability"] = rank_att
    return dataframe, pre_12, pre_13, pre_14



# raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\train\train_1.csv", encoding="utf-8", low_memory=False)
# print(raw_data["service_type"].value_counts())
# binary_value_distribution("service_type")
# scater_service_type("service_type")
# print(raw_data["is_mix_service"].value_counts())
# is_mix_service_1 = raw_data[raw_data["is_mix_service"] == 1]["current_service"]
# print(is_mix_service_1.value_counts())
# online_time_qu = raw_data[raw_data["online_time"] < 64]["service_type"]
# scater_service_type("online_time")
# bbox("online_time")
# print(online_time_qu.value_counts())
# scater_service_type("4_total_fee")
# bbox("4_total_fee")
# scater_service_type("month_traffic")
# bbox("month_traffic")
# month_traffic_3 = raw_data[raw_data["service_type"] == 3]["month_traffic"]
# print(month_traffic_3.value_counts())
# month_traffic_3 = raw_data[raw_data["service_type"] == 1]["month_traffic"]
# print(month_traffic_3.value_counts())

# binary_value_distribution("many_over_bill")
# binary_value_distribution("contract_type")
# scater_service_type("contract_type")
# scater_service_type("contract_time")
# binary_value_distribution("contract_time")
# binary_value_distribution("is_promise_low_consume")
# binary_value_distribution("net_service")
# scater_service_type("pay_times")
# scater_service_type("pay_num")
# bbox("pay_times")
# bbox("pay_num")
# temp = raw_data["pay_num"].tolist()
# error = [i for i in range(len(temp)) if temp[i]>40000]
# print(error)
# scater_service_type("last_month_traffic")
# binary_value_distribution("last_month_traffic")
# bbox("last_month_traffic")
# temp = raw_data["2_total_fee"].tolist()
# error = [i for i in range(len(temp)) if temp[i] == '\\N']
# float_fee_2 = [float(i) for i in temp]
# print([i for i in range(len(float_fee_2)) if float_fee_2[i] < 0])
# scater_service_type("2_total_fee")
# bbox("2_total_fee")
# temp = raw_data["3_total_fee"].tolist()
# error = [i for i in range(len(temp)) if temp[i] == '\\N']
# scater_service_type("3_total_fee")
# bbox("3_total_fee")
# scater_service_type("local_trafffic_month")
# bbox("local_trafffic_month")
# temp = raw_data["local_trafffic_month"].tolist()
# error = [i for i in range(len(temp)) if temp[i]>300000]
# print(error)
# scater_service_type("local_caller_time")
# bbox("local_caller_time")
# temp = raw_data["local_caller_time"].tolist()
# error = [i for i in range(len(temp)) if temp[i]>5000]
# print(error)
# binary_value_distribution("local_caller_time")
# scater_service_type("service1_caller_time")
# bbox("service1_caller_time")
# binary_value_distribution("service1_caller_time")
# scater_service_type("service2_caller_time")
# bbox("service2_caller_time")
# temp = raw_data["service2_caller_time"].tolist()
# error = [i for i in range(len(temp)) if temp[i]>8000]
# print(error)
# binary_value_distribution("service1_caller_time")
# temp = raw_data["gender"].tolist()
# error = [i for i in range(len(temp)) if temp[i] == '\\N']
# print(error)
# binary_value_distribution("gender")
# scater_service_type("age")
# temp = raw_data["age"].tolist()
# error = [i for i in range(len(temp)) if temp[i] == '\\N']
# print(error)
# bbox("age")
# binary_value_distribution("complaint_level")
# binary_value_distribution("former_complaint_num")
# bbox("former_complaint_num")
# scater_service_type("former_complaint_fee")
# bbox("former_complaint_fee")
# temp = raw_data["former_complaint_fee"].tolist()
# error = [i for i in range(len(temp)) if temp[i] > 10**9]
# print(error)
# binary_value_distribution("former_complaint_fee")
# dis_1 = raw_data[raw_data["service_type"] == 1]['former_complaint_fee'].tolist()
# dis_3 = raw_data[raw_data["service_type"] == 3]['former_complaint_fee'].tolist()
# dis_4 = raw_data[raw_data["service_type"] == 4]['former_complaint_fee'].tolist()
# sit_1 = [i for i in range(len(dis_1)) if ((dis_1[i] != 0) & (dis_1[i] < 10**10))]
# sit_3 = [i for i in range(len(dis_3)) if ((dis_3[i] != 0) & (dis_3[i] < 10**10))]
# sit_4 = [i for i in range(len(dis_4)) if ((dis_4[i] != 0) & (dis_4[i] < 10**10))]
# y_1 = [dis_1[i] for i in sit_1]
# y_3 = [dis_3[i] for i in sit_3]
# y_4 = [dis_4[i] for i in sit_4]
# print(np.mean(y_1))
# print(np.mean(y_3))
# print(np.mean(y_4))
# y = np.transpose(np.array([y_1, y_3, y_4]))
# labels = ["service_type_1", "service_type_3", "service_type_4"]
# plt.boxplot(y, labels=labels, sym='o')
# plt.grid(True)
# plt.show()
# print(binary_value_distribution("current_service"))


# temp = raw_data["1_total_fee"].tolist()
# error = [i for i in range(len(temp)) if temp[i]>4000]
# value = [temp[i] for i in error]
# print(value)

# temp = raw_data["month_traffic"].tolist()
# error = [i for i in range(len(temp)) if temp[i] > 120000]
# value = [temp[i] for i in error]
# print(value)
# bbox("contract_time")
# scater_service_type("pay_num")
# bbox("pay_num")
#

# raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\train\train_1.csv", encoding="utf-8",
#                        low_memory=False)
# binary_value_distribution("service_type")


# raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_sup_add_0_balance.csv", encoding="utf-8",
#                        low_memory=False)
# index = raw_data["current_service"].tolist()
# select_type = [12, 13, 14]
# index_num = len(index)
# select_data = [i for i in range(index_num) if index[i] in select_type]
# select_label = [index[i] for i in select_data]
# site_total = range(0, index_num)
# noisy_nom = round(0.3 * index_num)
# noisy = random.sample(site_total, noisy_nom)
# noisy_real = [i for i in noisy if i not in select_data]
# noisy_real_num = len(noisy_real)
# noisy_label = [0] * noisy_real_num
# select_total = select_data + noisy_real
# label_total = select_label + noisy_label
# select_sample = raw_data.iloc[select_total]
# select_sample["current_service_new"] = label_total
# select_sample.to_csv(r"E:\CCFDF\plansmatching\data\raw data\train\small_class_12_13_14_others.csv")
#

# b = np.mean([0.7064220183486238, 0.8772348033373063, 0.5764192139737991, 0.6644951140065146, 0.8432098765432099, 0.7623400365630713, 0.8747913188647747, 0.8597285067873304, 0.6574074074074074, 0.5882352941176471, 0.6779661016949153])
# print(b)
#
# r_3 = [0.7095435684647303, 0.8888888888888888, 0.7058823529411764, 0.6552901023890785, 0.8222778473091366, 0.7463837994214079, 0.8519195612431445, 0.8584269662921349, 0.6600331674958542, 0.611023622047244, 0.6757425742574258]
# print(np.mean(r_3))


raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_sup_add_0_correct.csv", encoding="utf-8",
                       low_memory=False)

attri_list = ['online_time_norm',
                  'local_trafffic_month_norm',
                  'service2_caller_time_norm', 'age_norm',
                  'fee_mean_norm', 'fee_mean_2_norm',
                  'fee_fluctuate_norm', 'month_traffic_norm', 'contract_time_norm', 'pay_num_norm',
                  'last_month_traffic_norm', 'local_trafffic_month_norm', 'local_caller_time_norm',
                  'service1_caller_time_norm']
parallel_coordinates(raw_data, attri_list)


# raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_sup_add_0_correct.csv", encoding="utf-8",
#                        low_memory=False)
# service_type = raw_data["current_service"].tolist()
# num_total = len(raw_data)
# select_type = [12, 13, 14]
# sel_site = [i for i in range(num_total) if service_type[i] in select_type]
# select_dataframe = raw_data.iloc[sel_site]
# select_dataframe.to_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_sup_add_12_13_14_correct.csv")
#
