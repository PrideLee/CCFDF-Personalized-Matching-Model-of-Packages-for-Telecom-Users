import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go



def scater_service_type(att):
    total_fee_s4_1 = raw_data[raw_data["service_type"] == 4][att].tolist()
    total_fee_s1_1 = raw_data[raw_data["service_type"] == 1][att].tolist()
    total_fee_s3_1 = raw_data[raw_data["service_type"] == 3][att].tolist()
    x_1 = [4+random.random() for i in range(len(total_fee_s4_1))]
    x_2 = [1+random.random() for i in range(len(total_fee_s1_1))]
    x_3 = [3+random.random() for i in range(len(total_fee_s3_1))]
    plt.scatter(x_1, total_fee_s4_1, c="red", s=0.03)
    plt.scatter(x_2, total_fee_s1_1, c="green", s=0.03)
    plt.scatter(x_3, total_fee_s3_1, c="blue", s=0.03)
    plt.xlabel('service_type')
    plt.ylabel(att)
    # plt.title("service_type-1_total_fee scatter")
    plt.show()


def box(att):
    y0 = raw_data[raw_data["service_type"] == 1][att].tolist()
    y1 = raw_data[raw_data["service_type"] == 3][att].tolist()
    y2 = raw_data[raw_data["service_type"] == 4][att].tolist()

    service_type_1 = go.Box(
        y=y0
    )
    service_type_3 = go.Box(
        y=y1
    )
    service_type_4 = go.Box(
        y=y2
    )
    data = [service_type_1, service_type_3, service_type_4]
    py.iplot(data)


def bbox(att):
    y0 = raw_data[raw_data["service_type"] == 1][att].tolist()
    y1 = raw_data[raw_data["service_type"] == 3][att].tolist()
    y2 = raw_data[raw_data["service_type"] == 4][att].tolist()
    y = np.transpose(np.array([y0, y1, y2]))
    labels = ["service_type_1", "service_type_3", "service_type_4"]
    plt.boxplot(y, labels=labels, sym='o')
    plt.grid(True)
    plt.show()


def binary_value_distribution(par):
    dis_1 = raw_data[raw_data["service_type"] == 1][par]
    dis_3 = raw_data[raw_data["service_type"] == 3][par]
    dis_4 = raw_data[raw_data["service_type"] == 4][par]
    print(dis_1.value_counts())
    print(dis_3.value_counts())
    print(dis_4.value_counts())


def dimensionality_reduction(par):
    new_list = []
    par_list = raw_data[par].tolist()
    for i in par_list:
        if i != 0:
            new_list.append(1)
        else:
            new_list.append(0)
    return new_list


# raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\train\train_1.csv", encoding="utf-8", low_memory=False)
# print(raw_data["service_type"].value_counts())
# print(raw_data["is_mix_service"].value_counts())
# is_mix_service_1 = raw_data[raw_data["is_mix_service"] == 1]["service_type"]
# print(is_mix_service_1.value_counts())
# online_time_qu = raw_data[raw_data["online_time"] < 64]["service_type"]
# print(online_time_qu.value_counts())
# scater_service_type("3_total_fee")
# bbox("3_total_fee")
# scater_service_type("month_traffic")
# bbox("month_traffic")
# month_traffic_3 = raw_data[raw_data["service_type"] == 3]["month_traffic"]
# print(month_traffic_3.value_counts())
# month_traffic_3 = raw_data[raw_data["service_type"] == 1]["month_traffic"]
# print(month_traffic_3.value_counts())
# binary_value_distribution("contract_type")
# binary_value_distribution("contract_time")
# binary_value_distribution("is_promise_low_consume")
# binary_value_distribution("net_service")
# scater_service_type("pay_times")
# scater_service_type("pay_num")
# box_plot("pay_times")
# bbox("pay_times")
# bbox("pay_num")
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
# scater_service_type("local_caller_time")
# bbox("local_caller_time")
# binary_value_distribution("local_caller_time")
# scater_service_type("service1_caller_time")
# bbox("service1_caller_time")
# binary_value_distribution("service1_caller_time")
# scater_service_type("service2_caller_time")
# bbox("service2_caller_time")
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
# scater_service_type("former_complaint_fee")
# bbox("former_complaint_fee")
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


