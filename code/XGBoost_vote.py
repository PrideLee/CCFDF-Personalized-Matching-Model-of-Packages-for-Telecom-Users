import pandas as pd
import numpy as np
import xgboost as xgb
from collections import Counter
import random
from sklearn.metrics import accuracy_score, confusion_matrix
# from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
# from sklearn import cross_validation, metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics



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



# raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\train4_final.csv",
#                        encoding="utf-8", low_memory=False)
raw_data = pd.read_csv(r"/data/projects/CCFDF_18/data/train4_final.csv",
                       encoding="utf-8", low_memory=False)
raw_data = raw_data.iloc[0:100]
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

para_list = ['is_mix_service', 'online_time', '1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee', 'fee_distance',
             '1_total_fee_norm', '2_total_fee_norm', '3_total_fee_norm', '4_total_fee_norm',
             'month_traffic', 'many_over_bill', 'contract_type', 'contract_time',
             'is_promise_low_consume', 'net_service', 'pay_times', 'pay_num', 'pay_mean', 'last_month_traffic',
             'local_trafffic_month', 'local_caller_time', 'service1_caller_time', 'service2_caller_time', 'gender',
             'age', 'complaint_level', 'former_complaint_num', 'former_complaint_fee',
             'fee_mean', 'fee_std', 'fee_fluctuate', 'fee_mean_2',
             'service_caller_time_fluctuate', 'service_caller_time_mean', 'online_time_norm', 'fee_mean_norm', 'fee_std_norm',
             'fee_fluctuate_norm', 'fee_distance_norm', 'month_traffic_norm', 'contract_time_norm', 'pay_num_norm',
             'last_month_traffic_norm', 'local_trafffic_month_norm', 'local_caller_time_norm',
             'service1_caller_time_norm', 'service2_caller_time_norm', 'age_norm', 'former_complaint_num_norm',
             'former_complaint_fee_norm', 'fee_mean_2_norm', 'service_caller_time_fluctuate_norm', 'service_caller_time_mean_norm',
             'month_traffic_precentage', 'contract_time_precentage',
             'pay_times_precentage', 'pay_num_precentage', 'last_month_traffic_precentage',
             'local_trafffic_month_precentage', 'local_caller_time_precentage', 'service1_caller_time_precentage',
             'service2_caller_time_precentage',
             'user_id']
label = raw_data["service_type_encode"].tolist()
par_list = para_list[:len(para_list) - 1]
select_data = raw_data[par_list]

# data_submit_raw = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\test4_final.csv",
#                               encoding="utf-8", low_memory=False)

data_submit_raw = pd.read_csv(r"/data/projects/CCFDF_18/data/test4_final.csv",
                               encoding="utf-8", low_memory=False)
data_submit = data_submit_raw[par_list]

submit_label_encode = []

F1_list = []
nrow = len(label)
for ixval in range(10):
    idxtest = [a for a in range(nrow) if a % 10 == ixval % 10]
    idxtrain = [a for a in range(nrow) if a % 10 != ixval % 10]
    label_train = [label[r] for r in idxtrain]
    label_test = [label[r] for r in idxtest]
    data_train = select_data.iloc[idxtrain]
    data_test = select_data.iloc[idxtest]

    # label_train, label_test, data_train, data_test = train_test_split(label, raw_data[par_list], test_size=0.02)

    m_class = xgb.XGBClassifier(learning_rate=0.1, n_estimators=1500, max_depth=7, min_child_weight=6, gamma=0,
                                subsample=0.8, n_jobs=-1, reg_alpha=0.05, reg_lambda=0.05,
                                colsample_bytree=0.8, objective='multi:softmax', num_class=8, seed=27)
    # 训练
    m_class.fit(data_train, label_train)
    test_8 = m_class.predict(data_test)
    print(test_8)
    print("Accuracy : %.2f" % accuracy_score(label_test, test_8))
    confusion_mat = confusion_matrix(label_test, test_8)
    print("Test confusion matrix")
    print(confusion_mat)
    F_sc = F1_score(confusion_mat)
    F1_list.append(F_sc)
    print("test F1_score")
    print(F_sc)

    submit_label_encode.append(m_class.predict(data_submit))
print("F1_list")
print(F1_list)
vote_combine = np.array(submit_label_encode).transpose()
vote_num = len(vote_combine)

vote_final = [Counter(vote_combine[i]).most_common(1)[0][0] for i in range(vote_num)]

decode_list = decode(vote_final)

user_id_4 = data_submit_raw["user_id"]

# submit_result = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\lgb_baseline.csv",
#                             encoding="utf-8", low_memory=False)
submit_result = pd.read_csv(r"/data/projects/CCFDF_18/data/lgb_baseline.csv", encoding="utf-8", low_memory=False)
origin_id = submit_result["user_id"].tolist()
origin_result = submit_result["current_service"].tolist()
num_4 = len(user_id_4)
for i in range(num_4):
    origin_result[origin_id.index(user_id_4[i])] = decode_list[i]
final_da = pd.DataFrame({"user_id": origin_id, "current_service": origin_result})
# final_da.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\XGBoost_vote.csv", index=False)
# final_da.to_csv(r"/data/projects/CCFDF_18/data/XGB_vote.csv", index=False)