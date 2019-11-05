import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split


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


# raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\train4_feature_2.csv",
#                        encoding="utf-8", low_memory=False)
raw_data = pd.read_csv(r"/data/projects/CCFDF_18/data/train4_feature_2.csv",
                       encoding="utf-8", low_memory=False)
# raw_data = raw_data.iloc[0:100]

# raw_data = pd.read_csv(r"/Users/peterlee/Documents/CCFDF18/final_data/class_2.csv",
#                        encoding="utf-8", low_memory=False)

# num_total = len(raw_data)
# random_site = random.sample(range(num_total), round(num_total*0.001))
# raw_data = raw_data.iloc[random_site]


# para_list = ['is_mix_service', 'online_time', '1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee',
#              'month_traffic', 'many_over_bill', 'contract_type', 'contract_time',
#              'is_promise_low_consume', 'net_service', 'pay_times', 'pay_num', 'last_month_traffic',
#              'local_trafffic_month', 'local_caller_time', 'service1_caller_time', 'service2_caller_time', 'gender',
#              'age', 'complaint_level', 'former_complaint_num', 'former_complaint_fee', 'user_id']

para_list = ['is_mix_service', 'online_time', '1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee',
             'fee_distance', 'fee_min', 'traffic_current_month',
             'month_traffic', 'many_over_bill', 'contract_type', 'contract_time',
             'is_promise_low_consume', 'net_service', 'pay_times', 'pay_num', 'last_month_traffic',
             'local_trafffic_month', 'local_caller_time', 'service1_caller_time', 'service2_caller_time', 'gender',
             'age', 'complaint_level', 'former_complaint_num', 'former_complaint_fee',
             'fee_mean', 'fee_std', 'fee_fluctuate', 'fee_mean_2',
             'service_caller_time_fluctuate', 'service_caller_time_mean',
             "fee_mat_subtract_01", "fee_mat_subtract_12", "fee_mat_subtract_23", "fee_mat_subtract_0112",
             "fee_mat_subtract_1223", "fee_mat_sum_01", "fee_mat_sum_12", "fee_mat_sum_23", "fee_mat_sum_0112",
             "pay_mean",  "month_traffic_per_fee", "contract_time_per_fee",
             "last_month_traffic_per_fee", "local_trafffic_month_per_fee", "local_caller_time_per_fee",
             "service_caller_time_mean_per_fee", "pay_fee_subtract", "pay_total", "pay_fee_total_subtract",
             'month_traffic_precentage', 'contract_time_precentage',
             'pay_times_precentage', 'pay_num_precentage', 'last_month_traffic_precentage',
             'local_trafffic_month_precentage', 'local_caller_time_precentage', 'service1_caller_time_precentage',
             'service2_caller_time_precentage',
             'user_id']

# para_list = ['fee_min', 'online_time_per_fee', '1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee',
#              'traffic_current_month', 'service_caller_time_fluctuate', 'online_time', 'month_traffic_per_fee',
#              'pay_fee_subtract', 'pay_fee_total_subtract', 'contract_time_per_fee', 'fee_fluctuate', 'month_traffic',
#              'service2_caller_time', 'service_caller_time_mean_per_fee', 'local_trafffic_month_per_fee',
#              'local_trafffic_month', 'age', 'fee_mean_2', 'last_month_traffic', 'local_caller_time', 'fee_mean',
#              'pay_num', 'fee_std', 'fee_distance',
#              'user_id']



label = raw_data["service_type_encode"].tolist()
par_list = para_list[:len(para_list) - 1]
label_train, label_test, data_train, data_test = train_test_split(label, raw_data[para_list], test_size=0.2)
data_test_id = data_test['user_id'].tolist()
data_train = data_train[par_list]
data_test = data_test[par_list]

m_class = xgb.XGBClassifier(learning_rate=0.1, n_estimators=1500, max_depth=7, min_child_weight=6, gamma=0,
                            subsample=0.8, n_jobs=-1, reg_alpha=0.05, reg_lambda=0.05,
                            colsample_bytree=0.8, objective='multi:softmax', num_class=8, seed=27)
# m_class = xgb.XGBClassifier(
#     max_depth=12, learning_rate=0.05, n_estimators=752, silent=True, objective='multi:softmax', gamma=0,
#     max_delta_step=0, subsample=1, colsample_bytree=0.9, colsample_bylevel=0.9, reg_alpha=1, reg_lambda=1,
#     scale_pos_weight=1, base_score=0.5, seed=2018, missing=None, num_class=8, n_jobs=-1)
# 训练
m_class.fit(data_train, label_train)
test_8 = m_class.predict(data_test)
print(test_8)
print("Accuracy : %.2f" % accuracy_score(label_test, test_8))
num_test = len(label_test)
error_id_list = [data_test_id[k] for k in range(num_test) if label_test[k] != test_8[k]]
pd.DataFrame({"error_id": error_id_list}).to_csv(
    r"/data/projects/CCFDF_18/data/XGB_del_error_id_2.csv", index=False)
confusion_mat = confusion_matrix(label_test, test_8)
print("Test confusion matrix")
print(confusion_mat)
F_sc = F1_score(confusion_mat)

print("test F1_score")
print(F_sc)

#
# plt.plot(eta_list, F1_list)
# plt.xlabel('Number of Trees in Ensemble')
# plt.ylabel('F1')
# plt.show()


# test_2 = m_class.predict_proba(X_test)
# 查看AUC评价标准

# 查看重要程度
attributes_name = np.array(par_list[:len(par_list)])
featureImportance = m_class.feature_importances_
idxSorted = np.argsort(featureImportance)

print("importance feature")
print(attributes_name[idxSorted])
print("feature score")
print(featureImportance[idxSorted])
##必须二分类才能计算
##print "AUC Score (Train): %f" % metrics.roc_auc_score(y_test, test_2)


# data_submit_raw = pd.read_csv(
#     r"E:\CCFDF\plansmatching\data\raw data\final_data\test4_feature_2.csv",
#     encoding="utf-8", low_memory=False)
data_submit_raw = pd.read_csv(
    r"/data/projects/CCFDF_18/data/test4_feature_2.csv",
    encoding="utf-8", low_memory=False)
data_submit = data_submit_raw[par_list]
submit_label_encode = m_class.predict(data_submit)
decode_list = decode(submit_label_encode)

user_id_4 = data_submit_raw["user_id"]

# submit_result = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\XGBoost_finalprodata.csv",
#                             encoding="utf-8", low_memory=False)
submit_result = pd.read_csv(r"/data/projects/CCFDF_18/data/XGBoost_finalprodata.csv",
                            encoding="utf-8", low_memory=False)
origin_id = submit_result["user_id"].tolist()
origin_result = submit_result["current_service"].tolist()
num_4 = len(user_id_4)
for i in range(num_4):
    origin_result[origin_id.index(user_id_4[i])] = decode_list[i]
final_da = pd.DataFrame({"user_id": origin_id, "current_service": origin_result})
# final_da.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\XGBoost_feature_del_open_modelpara.csv", index=False)
final_da.to_csv(r"/data/projects/CCFDF_18/data/XGBoost_feature_del2.csv", index=False)
