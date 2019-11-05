# 导入数据包
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import numpy as np
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


# 基础配置信息
# path = '/data/projects/CCFDF_18/data/'
path = 'E:\\CCFDF\\plansmatching\\data\\raw data\\final_data\\'


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
    "silent": False,
}

# 读取数据
train = pd.read_csv(path + 'train4_final.csv')
test = pd.read_csv(path + 'test4_final.csv')



# 小规模检验代码
# train = train.iloc[0:100]


# 构造原始数据
y = train.pop('service_type_encode')
train_id = train.pop('user_id')
# 这个字段有点问题
X = train

test_id = test['user_id']

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
             'service2_caller_time_precentage']


X = train[para_list]
X_test = test[para_list]






# 自定义F1评价函数
def f1_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    print(labels)
    preds = np.argmax(preds.reshape(8, -1), axis=0)
    score_vali = f1_score(y_true=labels, y_pred=preds, average='weighted')
    return 'f1_score', score_vali, True


xx_score = []
cv_pred = []

skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
for index, (train_index, test_index) in enumerate(skf.split(X, y)):
    print(train_index)
    X_train, X_valid, y_train, y_valid = X.iloc[train_index], X.iloc[test_index], y[train_index], y[test_index]

    train_data = lgb.Dataset(X_train, label=y_train)
    validation_data = lgb.Dataset(X_valid, label=y_valid)

    clf = lgb.train(params, train_data, num_boost_round=100000, valid_sets=[validation_data], early_stopping_rounds=50,
                     verbose_eval=1)

    xx_pred = clf.predict(X_valid, num_iteration=clf.best_iteration)

    xx_pred = [np.argmax(x) for x in xx_pred]

    xx_score.append(f1_score(y_valid, xx_pred, average='weighted'))

    y_test = clf.predict(X_test, num_iteration=clf.best_iteration)

    y_test = [np.argmax(x) for x in y_test]

    if index == 0:
        cv_pred = np.array(y_test).reshape(-1, 1)
    else:
        cv_pred = np.hstack((cv_pred, np.array(y_test).reshape(-1, 1)))

print(xx_score, np.mean(xx_score))
# 投票
submit = []
for line in cv_pred:
    submit.append(np.argmax(np.bincount(line)))


decode_list = decode(submit)
# submit_result_or = pd.read_csv(r"/data/projects/CCFDF_18/data/lgb_baseline.csv",
#                             encoding="utf-8", low_memory=False)

submit_result_or = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\lgb_baseline.csv",
                            encoding="utf-8", low_memory=False)

origin_id = submit_result_or["user_id"].tolist()
origin_result = submit_result_or["current_service"].tolist()
num_4 = len(test_id)
for i in range(num_4):
    origin_result[origin_id.index(test_id[i])] = decode_list[i]
# 保存结果
final_da = pd.DataFrame({"user_id": origin_id, "current_service": origin_result})
# print(final_da)
final_da.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\lgb_prodata.csv", index=False)

# final_da.to_csv(r"/data/projects/CCFDF_18/data/lgb_prodata.csv", index=False)
