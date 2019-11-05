# 导入数据包
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')


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



# 基础配置信息
# path = '/data/projects/CCFDF_18/data/'
# path = 'E:\\CCFDF\\plansmatching\\data\\raw data\\final_data\\'
path = 'E:\\CCFDF\\plansmatching\\data\\raw data\\final_data\\Chanle_B\\'
n_splits = 10
seed = 42

# lgb 参数
params = {
    "learning_rate": 0.1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.2,
    "max_depth": 4,
    "objective": "multiclass",
    "num_class": 15,
    "silent": True,
}

# 读取数据
# train = pd.read_csv(path + 'train4_feature_2.csv')
# test = pd.read_csv(path + 'test4_feature_2.csv')


train = pd.read_csv(path + 'train_4_feature_select.csv')
test = pd.read_csv(path + 'test_4_feature_select.csv')
# 小规模检验代码
# train = train.iloc[0:100]


# 构造原始数据
y = train.pop('service_type_encode')
# train_id = train.pop('user_id')
# 这个字段有点问题

test_id = test['user_id']

# para_list = ['is_mix_service', 'online_time', '1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee', 'fee_distance',
#              '1_total_fee_norm', '2_total_fee_norm', '3_total_fee_norm', '4_total_fee_norm',
#              'month_traffic', 'many_over_bill', 'contract_type', 'contract_time',
#              'is_promise_low_consume', 'net_service', 'pay_times', 'pay_num', 'pay_mean', 'last_month_traffic',
#              'local_trafffic_month', 'local_caller_time', 'service1_caller_time', 'service2_caller_time', 'gender',
#              'age', 'complaint_level', 'former_complaint_num', 'former_complaint_fee',
#              'fee_mean', 'fee_std', 'fee_fluctuate', 'fee_mean_2',
#              'service_caller_time_fluctuate', 'service_caller_time_mean', 'online_time_norm', 'fee_mean_norm', 'fee_std_norm',
#              'fee_fluctuate_norm', 'fee_distance_norm', 'month_traffic_norm', 'contract_time_norm', 'pay_num_norm',
#              'last_month_traffic_norm', 'local_trafffic_month_norm', 'local_caller_time_norm',
#              'service1_caller_time_norm', 'service2_caller_time_norm', 'age_norm', 'former_complaint_num_norm',
#              'former_complaint_fee_norm', 'fee_mean_2_norm', 'service_caller_time_fluctuate_norm', 'service_caller_time_mean_norm',
#              'month_traffic_precentage', 'contract_time_precentage',
#              'pay_times_precentage', 'pay_num_precentage', 'last_month_traffic_precentage',
#              'local_trafffic_month_precentage', 'local_caller_time_precentage', 'service1_caller_time_precentage',
#              'service2_caller_time_precentage']

# para_list = ['is_mix_service', 'online_time', '1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee',
#              'fee_distance', 'fee_min', 'traffic_current_month',
#              'month_traffic', 'many_over_bill', 'contract_type', 'contract_time',
#              'is_promise_low_consume', 'net_service', 'pay_times', 'pay_num', 'last_month_traffic',
#              'local_trafffic_month', 'local_caller_time', 'service1_caller_time', 'service2_caller_time', 'gender',
#              'age', 'complaint_level', 'former_complaint_num', 'former_complaint_fee',
#              'fee_mean', 'fee_std', 'fee_fluctuate', 'fee_mean_2',
#              'service_caller_time_fluctuate', 'service_caller_time_mean',
#              "fee_mat_subtract_01", "fee_mat_subtract_12", "fee_mat_subtract_23", "fee_mat_subtract_0112",
#              "fee_mat_subtract_1223", "fee_mat_sum_01", "fee_mat_sum_12", "fee_mat_sum_23", "fee_mat_sum_0112",
#              "pay_mean", "online_time_per_fee", "month_traffic_per_fee", "contract_time_per_fee",
#              "last_month_traffic_per_fee", "local_trafffic_month_per_fee", "local_caller_time_per_fee",
#              "service_caller_time_mean_per_fee", "pay_fee_subtract", "pay_total", "pay_fee_total_subtract",
#              'user_id']

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

# para_list = ['service_type', 'fee_min', 'traffic_current_month', 'contract_time', 'pay_total', 'local_caller_time',
#              'roaming_traffic', 'online_time', 'contract_type', 'fee_pay_subtract', 'service_caller_time_mean',
#              'age', 'age_hierarchy', 'user_id']

X = train[para_list]
X_test = test[para_list[:len(para_list) - 1]]



# 自定义F1评价函数
def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    print(labels)
    preds = np.argmax(preds.reshape(8, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='weighted')
    return 'f1_score', score_vali, True


y_train, y_valid, X_train, X_valid = train_test_split(y, X, test_size=0.03)

X_valid_id = X_valid['user_id'].tolist()
X_train = X_train[para_list[:len(para_list) - 1]]
X_valid = X_valid[para_list[:len(para_list) - 1]]

train_data = lgb.Dataset(X_train, label=y_train)
validation_data = lgb.Dataset(X_valid, label=y_valid)

clf = lgb.train(params, train_data, num_boost_round=100000, valid_sets=[validation_data], early_stopping_rounds=50,
                 verbose_eval=1)

xx_pred = clf.predict(X_valid, num_iteration=clf.best_iteration)
xx_pred = [np.argmax(x) for x in xx_pred]

print("Accuracy : %.2f" % accuracy_score(y_valid, xx_pred))
confusion_mat = confusion_matrix(y_valid, xx_pred)
print("Test confusion matrix")
print(confusion_mat)
F_sc = F1_score(confusion_mat)
print("test F1_score")
print(F_sc)

xx_score = f1_score(y_valid, xx_pred, average='weighted')
print(xx_score)
y_valid = y_valid.tolist()
num_valid = len(y_valid)
error_id = [X_valid_id[i] for i in range(num_valid) if y_valid[i] != xx_pred[i]]

# pd.DataFrame({'error_id': error_id}).to_csv(r"/data/projects/CCFDF_18/data/error_id_lgb.csv")

y_test = clf.predict(X_test, num_iteration=clf.best_iteration)
y_test = [np.argmax(x) for x in y_test]


decode_list = decode(y_test)
# submit_result_or = pd.read_csv(r"/data/projects/CCFDF_18/data/XGBoost_finalprodata.csv",
#                             encoding="utf-8", low_memory=False)

submit_result_or = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\XGBoost_finalprodata.csv",
                            encoding="utf-8", low_memory=False)

origin_id = submit_result_or["user_id"].tolist()
origin_result = submit_result_or["current_service"].tolist()

num_4 = len(test_id)
for i in range(num_4):
    origin_result[origin_id.index(test_id[i])] = decode_list[i]
# 保存结果
final_da = pd.DataFrame({"user_id": origin_id, "current_service": origin_result})
# print(final_da)
final_da.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\Chanle_B\lgb_feature_last.csv", index=False)

# final_da.to_csv(r"/data/projects/CCFDF_18/data/lgb_feature.csv", index=False)
