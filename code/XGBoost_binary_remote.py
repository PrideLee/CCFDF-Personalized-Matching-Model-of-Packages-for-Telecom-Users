import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
import xgboost as xgb

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


def type_final(bin, pro):
    num_train = len(bin)
    final_type = []
    for i in range(num_train):
        one_hot = bin.iloc[i].tolist()
        temp_site = [j for j in range(0, 7) if one_hot[j] == 0]
        if (len(temp_site) > 1) | (len(temp_site) == 0):
            pro_list = pro.iloc[i].tolist()
            final_type.append(pro_list.index(max(pro_list)))
        else:
            final_type.append(temp_site[0])
    return final_type


def clases_train(dataframe_raw, paralist, submit_dataframe):
    user_id_total = dataframe_raw["user_id"].tolist()
    service_type = dataframe_raw['service_type_encode'].tolist()
    dataframe_raw = dataframe_raw[paralist]
    # dataframe_raw = dataframe_raw.drop(["user_id"], axis=1)
    submit_dataframe = submit_dataframe[paralist]
    submit_dataframe_fin = submit_dataframe.drop(["user_id"], axis=1)
    num_total = len(service_type)
    prediction_type = []
    prediction_probability = []
    test_type = []
    test_probability = []
    label_train_set, label_test_set, data_train_set, data_test_set = train_test_split(service_type, dataframe_raw, test_size=0.05)
    data_test_set_fin = data_test_set.drop(['user_id'], axis=1)
    error_id = []
    # id_user = data_test["user_id"].tolist()
    # test_label_set = [service_type[user_id_total.index(y)] for y in id_user]
    for i in range(0, 8):
        type_positive_sample_temp_site = []
        type_negative_sample_temp_site = []
        re_service_type = []
        for j in range(num_total):
            if service_type[j] == i:
                type_positive_sample_temp_site.append(j)
                re_service_type.append(0)
            else:
                type_negative_sample_temp_site.append(j)
                re_service_type.append(1)
        # temp_positive_sample = service_type.iloc[type_positive_sample_temp_site]
        # temp_negative_sample = service_type.iloc[type_negative_sample_temp_site]
        # dataframe_raw['re_current_service'] = re_service_type
        # dataframe_new = dataframe_raw[paralist]
        label_train, label_test, data_train, data_test = train_test_split(re_service_type, dataframe_raw, test_size=0.05)

        temp_id = data_test['user_id'].tolist()
        # id_index_train = data_train["user_id"].tolist()
        # id_index_test = data_test["user_id"].tolist()
        data_train_fin = data_train.drop(['user_id'], axis=1)
        data_test_fin = data_test.drop(['user_id'], axis=1)

        m_class = xgb.XGBClassifier(learning_rate=0.15, n_estimators=1500, min_child_weight=7, max_depth=6, gamma=0,
                                    subsample=0.8, n_jobs=-1, colsample_bytree=0.8, objective='binary:logistic',
                                    reg_alpha=0.05, reg_lambda=0.05, seed=0)

        m_class.fit(data_train_fin, label_train)
        # Accumulate auc on test set
        prediction = m_class.predict(data_test_fin)
        correct = accuracy_score(label_test, prediction)
        prediction_train = m_class.predict(data_train_fin)
        correct_train = accuracy_score(label_train, prediction_train)
        label_test_num = len(data_test_fin)
        error_id.append([temp_id[y] for y in range(label_test_num) if label_test[y] != prediction[y]])
        print("train correct")
        print(correct_train)
        # generate confusion matrix
        pList = prediction.tolist()
        confusionMat = confusion_matrix(label_test, pList)
        print("F1 Score test")
        print(F1_score(confusionMat))

        prediction_test = m_class.predict(data_test_set_fin)
        test_type.append(prediction_test)
        test_pro_temp = m_class.predict_proba(data_test_set_fin)
        num_test = len(data_test_set_fin)
        test_probability.append([test_pro_temp[p][0] for p in range(num_test)])

        prediction_type.append(m_class.predict(submit_dataframe_fin))
        pro_temp = m_class.predict_proba(submit_dataframe_fin)
        num_sub = len(submit_dataframe_fin)
        prediction_probability.append([pro_temp[v][0] for v in range(num_sub)])

    test_type_array = np.array(test_type).transpose()
    test_probability_array = np.array(test_probability).transpose()
    test_type_df = pd.DataFrame(test_type_array)
    test_probability_df = pd.DataFrame(test_probability_array)
    test_type_final = type_final(test_type_df, test_probability_df)

    confusionMat_test = confusion_matrix(label_test_set, test_type_final)
    print(label_test_set)
    print(test_type_final)
    f1_test = F1_score(confusionMat_test)
    print("F1_test")
    print(f1_test)

    prediction_type_array = np.array(prediction_type).transpose()
    prediction_probability_array = np.array(prediction_probability).transpose()
    prediction_type_df = pd.DataFrame(prediction_type_array)
    prediction_probability_df = pd.DataFrame(prediction_probability_array)

    pd.DataFrame({'id_error':error_id}).to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\XGBoost_binary_error_id_1")
    # pd.DataFrame({'id_error': error_id}).to_csv(r"/data/projects/CCFDF_18/data/XGBoost_binary_error_id")
    return prediction_type_df, prediction_probability_df



raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\train4_final.csv",
                       encoding="utf-8", low_memory=False)
# raw_data = pd.read_csv(r"/data/projects/CCFDF_18/data/train4_final.csv",
#                        encoding="utf-8", low_memory=False)
# submit_data = pd.read_csv(r"/Users/peterlee/Documents/CCFDF18/final_data/.csv",
#                        encoding="utf-8", low_memory=False)
# submit_data = submit_data.iloc[0:100]

para_list = ['is_mix_service', 'online_time', '1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee', 'fee_distance',
             '1_total_fee_norm', '2_total_fee_norm', '3_total_fee_norm', '4_total_fee_norm',
             'month_traffic', 'many_over_bill', 'contract_type', 'contract_time',
             'is_promise_low_consume', 'net_service', 'pay_times', 'pay_num', 'pay_mean','last_month_traffic',
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


# sub_data = pd.read_csv(r"/data/projects/CCFDF_18/data/test4_final.csv",
#         encoding="utf-8", low_memory=False)
sub_data = pd.read_csv(
    r"E:\CCFDF\plansmatching\data\raw data\final_data\test4_final.csv",
    encoding="utf-8", low_memory=False)
type_binary, pro_binary = clases_train(raw_data, para_list, sub_data)
type_sub = type_final(type_binary, pro_binary)
decode_list = decode(type_sub)
user_id_4 = sub_data["user_id"]
submit_result = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\lgb_baseline.csv",
                            encoding="utf-8", low_memory=False)
# submit_result = pd.read_csv(r"/data/projects/CCFDF_18/data/lgb_baseline.csv", encoding="utf-8", low_memory=False)
origin_id = submit_result["user_id"].tolist()
origin_result = submit_result["current_service"].tolist()
num_4 = len(user_id_4)
for i in range(num_4):
    origin_result[origin_id.index(user_id_4[i])] = decode_list[i]
final_da = pd.DataFrame({"user_id": origin_id, "current_service": origin_result})
# final_da.to_csv(r"/data/projects/CCFDF_18/data/XGBoost_binary_1.csv")

final_da.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\XGBoost_binary_2.csv")












