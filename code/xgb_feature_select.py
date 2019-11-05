import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
# import matplotlib.pylab as plt
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE

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


raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\train4_feature_2.csv",
                       encoding="utf-8", low_memory=False)

# raw_data = pd.read_csv(r"/data/projects/CCFDF_18/data/train4_feature_2.csv",
#                        encoding="utf-8", low_memory=False)

# raw_data = raw_data.iloc[0:100]

# raw_data = pd.read_csv(r"/Users/peterlee/Documents/CCFDF18/final_data/class_2.csv",
#                        encoding="utf-8", low_memory=False)

# num_total = len(raw_data)
# random_site = random.sample(range(num_total), round(num_total*0.001))
# raw_data = raw_data.iloc[random_site]


para_list = ['is_mix_service', 'online_time', '1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee',
             'fee_distance',
             '1_total_fee_norm', '2_total_fee_norm', '3_total_fee_norm', '4_total_fee_norm',
             'month_traffic', 'many_over_bill', 'contract_type', 'contract_time',
             'is_promise_low_consume', 'net_service', 'pay_times', 'pay_num', 'last_month_traffic',
             'local_trafffic_month', 'local_caller_time', 'service1_caller_time', 'service2_caller_time', 'gender',
             'age', 'complaint_level', 'former_complaint_num', 'former_complaint_fee',
             'fee_mean', 'fee_std', 'fee_fluctuate', 'fee_mean_2',
             'service_caller_time_fluctuate', 'service_caller_time_mean', 'online_time_norm', 'fee_mean_norm',
             'fee_std_norm',
             'fee_fluctuate_norm', 'fee_distance_norm', 'month_traffic_norm', 'contract_time_norm', 'pay_num_norm',
             'last_month_traffic_norm', 'local_trafffic_month_norm', 'local_caller_time_norm',
             'service1_caller_time_norm', 'service2_caller_time_norm', 'age_norm', 'former_complaint_num_norm',
             'former_complaint_fee_norm', 'fee_mean_2_norm', 'service_caller_time_fluctuate_norm',
             'service_caller_time_mean_norm',
             'month_traffic_precentage', 'contract_time_precentage',
             'pay_times_precentage', 'pay_num_precentage', 'last_month_traffic_precentage',
             'local_trafffic_month_precentage', 'local_caller_time_precentage', 'service1_caller_time_precentage',
             'service2_caller_time_precentage',
             "fee_mat_subtract_01", "fee_mat_subtract_12", "fee_mat_subtract_23", "fee_mat_subtract_0112",
             "fee_mat_subtract_1223", "fee_mat_sum_01", "fee_mat_sum_12", "fee_mat_sum_23", "fee_mat_sum_0112",
             "pay_mean", "online_time_per_fee", "month_traffic_per_fee", "contract_time_per_fee",
             "last_month_traffic_per_fee", "local_trafffic_month_per_fee", "local_caller_time_per_fee",
             "service_caller_time_mean_per_fee", "pay_fee_subtract", "pay_total", "pay_fee_total_subtract",
             "fee_mat_subtract_01_norm", 'fee_min', 'traffic_current_month',
             "fee_mat_subtract_12_norm", "fee_mat_subtract_23_norm", "fee_mat_subtract_0112_norm",
             "fee_mat_subtract_1223_norm", "fee_mat_sum_01_norm", "fee_mat_sum_12_norm", "fee_mat_sum_23_norm",
             "fee_mat_sum_0112_norm", "pay_mean_norm", "online_time_per_fee_norm", "month_traffic_per_fee_norm",
             "contract_time_per_fee_norm",
             "last_month_traffic_per_fee_norm", "local_trafffic_month_per_fee_norm", "local_caller_time_per_fee_norm",
             "service_caller_time_mean_per_fee_norm", "pay_fee_subtract_norm", "pay_total_norm",
             "pay_fee_total_subtract_norm",
             'user_id']

# para_list = ['is_mix_service', 'online_time', '1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee',
#                  '1_total_fee_log', '2_total_fee_log', '3_total_fee_log', '4_total_fee_log',
#                  'month_traffic', 'many_over_bill', 'contract_type', 'contract_time',
#                  'is_promise_low_consume', 'net_service', 'pay_times', 'pay_num', 'last_month_traffic',
#                  'local_trafffic_month', 'local_caller_time', 'service1_caller_time', 'service2_caller_time', 'gender',
#                  'age', 'complaint_level', 'former_complaint_num', 'former_complaint_fee',
#                  'fee_mean', 'fee_std', 'fee_fluctuate', 'fee_mean_2', 'fee_min', 'traffic_current_month',
#                  'service_caller_time_fluctuate', 'online_time_log', 'fee_mean_log', 'fee_std_log',
#                  'fee_fluctuate_log', 'month_traffic_log', 'contract_time_log', 'pay_num_log',
#                  'last_month_traffic_log', 'local_trafffic_month_log', 'local_caller_time_log',
#                  'service1_caller_time_log', 'service2_caller_time_log', 'age_log', 'former_complaint_num_log',
#                  'former_complaint_fee_log', 'fee_mean_2_log', 'service_caller_time_fluctuate_log',
#                  'month_traffic_precentage', 'contract_time_precentage',
#                  'pay_times_precentage', 'pay_num_precentage', 'last_month_traffic_precentage',
#                  'local_trafffic_month_precentage', 'local_caller_time_precentage', 'service1_caller_time_precentage',
#                  'service2_caller_time_precentage', 'user_id']


# para_list = ['is_mix_service', 'online_time', '1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee',
#                  '1_total_fee_norm', '2_total_fee_norm', '3_total_fee_norm', '4_total_fee_norm',
#                  'month_traffic', 'many_over_bill', 'contract_type', 'contract_time', 'fee_min',
#                  'is_promise_low_consume', 'net_service', 'pay_times', 'pay_num', 'last_month_traffic',
#                  'local_trafffic_month', 'local_caller_time', 'service1_caller_time', 'service2_caller_time', 'gender',
#                  'age', 'complaint_level', 'former_complaint_num', 'former_complaint_fee',
#                  'fee_mean', 'fee_std', 'fee_fluctuate', 'fee_mean_2',
#                  'service_caller_time_fluctuate', 'online_time_norm', 'fee_mean_norm', 'fee_std_norm',
#                  'fee_fluctuate_norm', 'month_traffic_norm', 'contract_time_norm', 'pay_num_norm',
#                  'last_month_traffic_norm', 'local_trafffic_month_norm', 'local_caller_time_norm',
#                  'service1_caller_time_norm', 'service2_caller_time_norm', 'age_norm', 'former_complaint_num_norm',
#                  'former_complaint_fee_norm', 'fee_mean_2_norm', 'service_caller_time_fluctuate_norm', 'traffic_current_month',
#                  'user_id']

label = raw_data["service_type_encode"].tolist()
par_list = para_list[:len(para_list) - 1]
label_train, label_test, data_train, data_test = train_test_split(label, raw_data[par_list], test_size=0.03)

estimator = xgb.XGBClassifier(learning_rate=0, n_estimators=1500, max_depth=7, min_child_weight=6, gamma=0,
                              subsample=0.8, n_jobs=-1, reg_alpha=0.05, reg_lambda=0.05,
                              colsample_bytree=0.8, objective='multi:softmax', num_class=8, seed=2018)

# selector = RFECV(estimator, step=10, cv=3, n_jobs=-1)
selector = RFE(estimator=estimator, step=10, n_features_to_select=50)
selector.fit(data_train, label_train)



print("N_features %s" % selector.n_features_) # 保留的特征数
print("Support is %s" % selector.support_) # 是否保留
print("Ranking %s" % selector.ranking_) # 重要程度排名
# print("Grid Scores %s" % selector.grid_scores_)

# Plot number of features VS. cross-validation scores
# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross validation score (nb of correct classifications")
# plt.plot(range(1, len(RFECV.grid_scores_) + 1), RFECV.grid_scores_)
# plt.show()

# print("Number of features selected")
# print([i for i in range(1, len(selector.grid_scores_) + 1)])
# print("Cross validation score (nb of correct classifications")
# print(selector.grid_scores_)

test_8 = selector.predict(data_test)
print(test_8)
print("Accuracy : %.2f" % accuracy_score(label_test, test_8))
confusion_mat = confusion_matrix(label_test, test_8)
print("Test confusion matrix")
print(confusion_mat)
F_sc = F1_score(confusion_mat)
print("test F1_score")
print(F_sc)



# 查看重要程度
# attributes_name = np.array(par_list[:len(par_list)])
# featureImportance = selector.feature_importances_
# idxSorted = np.argsort(featureImportance)
# print("importance feature")
# print(attributes_name[idxSorted])
# print("feature score")
# print(featureImportance[idxSorted])
# barPos = np.arange(idxSorted.shape[0]) + .5
# plt.barh(barPos, featureImportance[idxSorted], align='center')
# plt.yticks(barPos, attributes_name[idxSorted])
# plt.xlabel('Variable Importance')
# plt.show()

# data_submit_raw = pd.read_csv(
#     r"/data/projects/CCFDF_18/data/test4_feature_2.csv",
#     encoding="utf-8", low_memory=False)

data_submit_raw = pd.read_csv(
    r"E:\CCFDF\plansmatching\data\raw data\final_data\test4_feature_2.csv",
    encoding="utf-8", low_memory=False)
data_submit = data_submit_raw[par_list]
submit_label_encode = selector.predict(data_submit)
decode_list = decode(submit_label_encode)

user_id_4 = data_submit_raw["user_id"]

# submit_result = pd.read_csv(r"/data/projects/CCFDF_18/data/XGBoost_finalprodata.csv",
#                             encoding="utf-8", low_memory=False)

submit_result = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\XGBoost_finalprodata.csv",
                            encoding="utf-8", low_memory=False)
origin_id = submit_result["user_id"].tolist()
origin_result = submit_result["current_service"].tolist()
num_4 = len(user_id_4)
for i in range(num_4):
    origin_result[origin_id.index(user_id_4[i])] = decode_list[i]
final_da = pd.DataFrame({"user_id": origin_id, "current_service": origin_result})
final_da.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\XGBoost_feature_rfesel.csv", index=False)
# final_da.to_csv(r"/data/projects/CCFDF_18/data/XGBoost_feature_cvsel.csv", index=False)