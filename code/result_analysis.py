import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


def f1_every(matrix_confusion):
    precision = []
    recall = []
    F1 = []
    class_num = len(matrix_confusion)
    for i in range(class_num):
        temp_row = matrix_confusion[i]
        TP = temp_row[i]
        FN_sum = sum(temp_row)
        temp_column = matrix_confusion[:, i]
        FP_sum = sum(temp_column)
        pre = TP / max(FP_sum, 1)
        rec = TP / max(FN_sum, 1)
        f1 = (2 * pre * rec) / max((pre + rec), 1)
        F1.append(f1)
        precision.append(pre)
        recall.append(rec)
    return F1, precision, recall


def scatter(attr, dataframe_file):
    data = dataframe_file[attr].tolist()
    num = len(data)
    x_1 = [1 + random.uniform(0, 3) for i in range(num)]
    plt.scatter(x_1, data, c="red", s=0.3)
    plt.show()


def box(attr, dataframe_file):
    data = dataframe_file[attr].tolist()
    labels = ['0']
    plt.boxplot(data, labels=labels, sym='o')
    plt.grid(True)
    plt.show()


#
# raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_sup.csv", encoding="utf-8",
#                        low_memory=False)

# error_id_0 = pd.read_csv(r"E:\CCFDF\plansmatching\result\class_2\RF\error_id_0.csv",  encoding="utf-8")
# error_id_0_list = error_id_0["error_id"].tolist()
# error_id_1 = pd.read_csv(r"E:\CCFDF\plansmatching\result\class_2\RF\error_id_1.csv",  encoding="utf-8")
# error_id_1_list = error_id_0["error_id"].tolist()
# error_id_2 = pd.read_csv(r"E:\CCFDF\plansmatching\result\class_2\RF\error_id_2.csv",  encoding="utf-8")
# error_id_2_list = error_id_0["error_id"].tolist()
# error_id_3 = pd.read_csv(r"E:\CCFDF\plansmatching\result\class_2\RF\error_id_3.csv",  encoding="utf-8")
# error_id_3_list = error_id_0["error_id"].tolist()
# error_id_4 = pd.read_csv(r"E:\CCFDF\plansmatching\result\class_2\RF\error_id_4.csv",  encoding="utf-8")
# error_id_4_list = error_id_4["error_id"].tolist()
# error_id_list_total = error_id_0_list + error_id_1_list + error_id_2_list + error_id_3_list + error_id_4_list
# site = raw_data["user_id"].tolist()
# site_list = [site.index(i) for i in error_id_list_total]
# error_dataframe = raw_data.iloc[site_list]


# error_id_balance_1 = pd.read_csv(r"E:\CCFDF\plansmatching\result\class_2\RF\error_id_5_balance_1.csv",  encoding="utf-8")
# error_id_balance_1_list = error_id_balance_1["error_id"].tolist()
# site = raw_data["user_id"].tolist()
# site_list = [site.index(i) for i in error_id_balance_1_list]
# error_dataframe = raw_data.iloc[site_list]
# error_dataframe.to_csv(r"E:\CCFDF\plansmatching\result\class_2\RF\error_dataframe_balance_1.csv")
# print(error_dataframe["current_service"].value_counts())

error_data_dataframe = pd.read_csv(r"E:\CCFDF\plansmatching\result\class_2\RF\error_dataframe_balance_1.csv",
                                   encoding="utf-8", low_memory=False)
select_attr = ['online_time_norm', 'month_traffic_norm', 'many_over_bill', 'contract_type', 'contract_time_norm',
               'pay_times', 'pay_num_norm', 'last_month_traffic_norm', 'local_trafffic_month_norm',
               'local_caller_time_norm',
               'service1_caller_time_norm', 'service2_caller_time_norm', 'gender', 'age_norm', 'fee_mean', 'fee_mean_2',
               'fee_std', 'fee_mean_2_norm',
               'fee_fluctuate', 'current_service']
error_data_dataframe_select = error_data_dataframe[select_attr]
confusion_mat = pd.read_csv(r"E:\CCFDF\plansmatching\result\class_2\RF\confusion_matrix.csv", encoding="utf-8")
row_num = len(confusion_mat)
matrix_confusion = np.array(confusion_mat.tail(11))
f1_sc, pre, rec = f1_every(matrix_confusion)
# print(f1_sc)
# print(pre)
# print(rec)
# focus = [i for i in range(len(f1_sc)) if f1_sc[i] < 0.86]
# print(focus)
error_14 = error_data_dataframe_select[error_data_dataframe_select["current_service"] == 14]
scatter("month_traffic_norm", error_14)
box("month_traffic_norm", error_14)

# error_dataframe.to_csv(r"E:\CCFDF\plansmatching\result\class_2\RF\error_dataframe_total.csv")
# error_data_dataframe = pd.read_csv(r"E:\CCFDF\plansmatching\result\class_2\RF\error_dataframe_0.csv", encoding="utf-8",
#                                    low_memory=False)
