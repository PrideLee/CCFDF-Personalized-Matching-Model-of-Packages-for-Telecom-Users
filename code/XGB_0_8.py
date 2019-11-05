import pandas as pd
import numpy as np
import random
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
from sklearn.externals import joblib


def data_balance(file_dataframe, balance_type, propotion_del):
    type_list = file_dataframe["service_type_encode"].tolist()
    site_sample = []
    for i in range(len(balance_type)):
        site = [j for j in range(len(type_list)) if type_list[j] == balance_type[i]]
        num = round(len(site) * propotion_del[i])
        site_sample += random.sample(site, num)
    site_total = [k for k in range(len(type_list))]
    for m in site_sample:
        site_total.remove(m)
    balance_data = file_dataframe.iloc[site_total]
    return balance_data


def up_sampleing(file_dataframe, balance_type, times):
    type_list = file_dataframe["service_type_encode"].tolist()
    up_sam = []
    for i in range(len(balance_type)):
        site = [j for j in range(len(type_list)) if type_list[j] == balance_type[i]]
        up_sam += site * times[i]
    dataframe = pd.concat([file_dataframe, file_dataframe.iloc[up_sam]])
    return dataframe


def F1_score(confusion_max):
    precision = []
    recall = []
    F1 = []
    class_num = len(confusion_max)
    for i in range(class_num):
        temp_row = confusion_max[i]
        TP = temp_row[i]
        FN_sum = sum(temp_row)
        temp_column = confusion_max[:, i]
        FP_sum = sum(temp_column)
        pre = TP / max(FP_sum, 1)
        rec = TP / max(FN_sum, 1)
        f1 = (2 * pre * rec) / max((pre + rec), 1)
        F1.append(f1)
        precision.append(pre)
        recall.append(rec)
    print("F1")
    print(F1)
    print("precision")
    print(precision)
    print("recall")
    print(recall)
    F_score = ((1 / len(F1)) * sum(F1)) ** 2
    return F_score


def decode(encode_list):
    final_re = []
    for i in encode_list:
        if i == 0:
            final_re.append(89950166)
        if i == 1:
            final_re.append(89950167)
        if i == 2:
            final_re.append(89950168)
        if i == 3:
            final_re.append(99999825)
        if i == 4:
            final_re.append(99999826)
        if i == 5:
            final_re.append(99999827)
        if i == 6:
            final_re.append(99999828)
        if i == 7:
            final_re.append(99999830)
    return final_re


raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\Chanle_B\train_4_feature_select.csv",
                       encoding="utf-8", low_memory=False)
type_total = raw_data["service_type_encode"].tolist()

num_total = len(type_total)
index_0_8 = [i for i in range(num_total) if (type_total[i] == 0) or (type_total[i] == 7)]
select_data = raw_data.iloc[index_0_8]
# select_data_bal = data_balance(select_data, [0], [0.5])
select_data_bal = up_sampleing(select_data, [7], [3])
# para_list = ['service_type', 'fee_min', 'traffic_current_month', 'contract_time', 'pay_total', 'local_caller_time',
#              'roaming_traffic', 'online_time', 'contract_type', 'fee_pay_subtract', 'service_caller_time_mean',
#              'age', 'age_hierarchy', 'user_id']

para_list = ['online_time', '1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee',
             'fee_distance',
             'month_traffic', 'many_over_bill', 'contract_type', 'contract_time', 'roaming_traffic', 'fee_pay_subtract',
             'pay_num', 'last_month_traffic', 'local_trafffic_month', 'local_caller_time', 'service1_caller_time',
             'service2_caller_time', 'age', 'fee_mean', 'fee_std', 'fee_fluctuate', 'fee_mean_2',
             'service_caller_time_fluctuate', 'service_caller_time_mean',
             'fee_mat_subtract_01', 'fee_mat_subtract_12', 'fee_mat_subtract_23', 'fee_mat_sum_01', 'fee_mat_sum_12',
             'fee_mat_sum_23', 'fee_mat_sum_0112', 'pay_mean', 'online_time_per_fee', 'month_traffic_per_fee',
             'contract_time_per_fee', 'last_month_traffic_per_fee', 'local_trafffic_month_per_fee',
             'service_caller_time_mean_per_fee', 'pay_fee_subtract', 'pay_total', 'fee_min', 'traffic_current_month',
             'user_id']

# select_data = select_data.iloc[0:1000]
label_bl = select_data_bal["service_type_encode"].tolist()

par_list = para_list[:len(para_list) - 1]

label_train, label_test, data_train, data_test = train_test_split(label_bl, select_data_bal[par_list], test_size=0.05)

m_class = xgb.XGBClassifier(learning_rate=0.1, n_estimators=1500, max_depth=7, min_child_weight=6, gamma=0,
                            subsample=0.8, n_jobs=-1, reg_alpha=0.05, reg_lambda=0.05,
                            colsample_bytree=0.8, objective='binary:logistic', seed=27)
# 训练
m_class.fit(data_train, label_train)
test_8 = m_class.predict(data_test)
print("Accuracy : %.2f" % accuracy_score(label_test, test_8))
confusion_mat = confusion_matrix(label_test, test_8)
print("Test confusion matrix")
print(confusion_mat)
F_sc = F1_score(confusion_mat)
print("test F1_score")
print(F_sc)

label = select_data["service_type_encode"].tolist()
label_train, label_test, data_train, data_test = train_test_split(label, select_data[par_list], test_size=0.2)

test_8 = m_class.predict(data_test)
print("Accuracy : %.2f" % accuracy_score(label_test, test_8))
confusion_mat = confusion_matrix(label_test, test_8)
print("Test confusion matrix")
print(confusion_mat)
F_sc = F1_score(confusion_mat)
print("test F1_score")
print(F_sc)


# test_2 = m_class.predict_proba(X_test)
# 查看AUC评价标准

# 查看重要程度
attributes_name = np.array(par_list[:len(par_list)])
featureImportance = m_class.feature_importances_
idxSorted = np.argsort(featureImportance)
barPos = np.arange(idxSorted.shape[0]) + .5
plt.barh(barPos, featureImportance[idxSorted], align='center')
plt.yticks(barPos, attributes_name[idxSorted])
plt.xlabel('Variable Importance')
plt.show()
print("importance feature")
print(attributes_name[idxSorted])
print("feature score")
print(featureImportance[idxSorted])

# m_class.save_model(r'C:\Users\poder\Desktop\xgb_0_8.model')
joblib.dump(m_class, r'E:\CCFDF\plansmatching\data\raw data\final_data\Chanle_B\xgb_0_8_up_real_1.model')

