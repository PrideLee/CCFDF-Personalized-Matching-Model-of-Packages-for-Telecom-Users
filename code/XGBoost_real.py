import pandas as pd
import numpy as np
import xgboost as xgb
import random
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
# from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import cross_validation, metrics
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pylab as plt
from sklearn import metrics
from matplotlib.pylab import rcParams


if __name__ == '__main__':

    rcParams['figure.figsize'] = 12, 4


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



    raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\train_combine_4_encode_precentage.csv",
                           encoding="utf-8", low_memory=False)

    # raw_data = data_balance(raw_data, [0, 1, 6], [0.7, 0.4, 0.2])
    # num_total = len(raw_data)
    # random_site = random.sample(range(num_total), round(num_total*0.3))
    # raw_data = raw_data.iloc[random_site]
    #
    # para_list = ['is_mix_service',
    #              '1_total_fee_norm', '2_total_fee_norm', '3_total_fee_norm', '4_total_fee_norm',
    #              'many_over_bill', 'contract_type',
    #              'is_promise_low_consume', 'net_service', 'pay_times',
    #              'gender',
    #              'complaint_level',
    #              'online_time_norm', 'fee_mean_norm', 'fee_std_norm',
    #              'fee_fluctuate_norm', 'month_traffic_norm', 'contract_time_norm', 'pay_num_norm',
    #              'last_month_traffic_norm', 'local_trafffic_month_norm', 'local_caller_time_norm',
    #              'service1_caller_time_norm', 'service2_caller_time_norm', 'age_norm', 'former_complaint_num_norm',
    #              'former_complaint_fee_norm', 'fee_mean_2_norm', 'service_caller_time_fluctuate_norm',
    #              'month_traffic_precentage', 'contract_time_precentage',
    #              'pay_times_precentage', 'pay_num_precentage', 'last_month_traffic_precentage',
    #              'local_trafffic_month_precentage', 'local_caller_time_precentage', 'service1_caller_time_precentage',
    #              'service2_caller_time_precentage',
    #              'user_id'
    #              ]

    # par_list = ['month_traffic_norm', '3_total_fee_norm', '1_total_fee_norm', 'fee_std_norm', '2_total_fee_norm',
    #             'service2_caller_time_norm', 'online_time_norm', 'local_trafffic_month_norm', '4_total_fee_norm',
    #             'fee_fluctuate_norm', 'fee_mean_2_norm', 'service_caller_time_fluctuate_norm', 'local_caller_time_norm',
    #             'fee_mean_norm', 'last_month_traffic_norm', 'age_norm', 'pay_num_norm', 'contract_type',
    #             'contract_time_norm', 'service1_caller_time_norm']



    para_list = ['is_mix_service', 'online_time', '1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee',
                 '1_total_fee_norm', '2_total_fee_norm', '3_total_fee_norm', '4_total_fee_norm',
                 'month_traffic', 'many_over_bill', 'contract_type', 'contract_time',
                 'is_promise_low_consume', 'net_service', 'pay_times', 'pay_num', 'last_month_traffic',
                 'local_trafffic_month', 'local_caller_time', 'service1_caller_time', 'service2_caller_time', 'gender',
                 'age', 'complaint_level', 'former_complaint_num', 'former_complaint_fee',
                 'fee_mean', 'fee_std', 'fee_fluctuate', 'fee_mean_2',
                 'service_caller_time_fluctuate', 'online_time_norm', 'fee_mean_norm', 'fee_std_norm',
                 'fee_fluctuate_norm', 'month_traffic_norm', 'contract_time_norm', 'pay_num_norm',
                 'last_month_traffic_norm', 'local_trafffic_month_norm', 'local_caller_time_norm',
                 'service1_caller_time_norm', 'service2_caller_time_norm', 'age_norm', 'former_complaint_num_norm',
                 'former_complaint_fee_norm', 'fee_mean_2_norm', 'service_caller_time_fluctuate_norm',
                 'month_traffic_precentage', 'contract_time_precentage',
                 'pay_times_precentage', 'pay_num_precentage', 'last_month_traffic_precentage',
                 'local_trafffic_month_precentage', 'local_caller_time_precentage', 'service1_caller_time_precentage',
                 'service2_caller_time_precentage', 'user_id']


    # para_list = ['1_total_fee', '3_total_fee', 'month_traffic', '2_total_fee', 'online_time', '4_total_fee',
    #              'service2_caller_time', 'last_month_traffic', 'local_trafffic_month', 'fee_std',
    #              'service_caller_time_fluctuate', 'fee_fluctuate', 'fee_mean_2', 'pay_num', 'local_caller_time',
    #              'fee_mean', 'age', 'contract_type', 'contract_time', 'service1_caller_time', '1_total_fee_norm',
    #              '2_total_fee_norm', '3_total_fee_norm', '4_total_fee_norm', 'user_id']
    select_data = raw_data[para_list]
    label = raw_data["service_type_encode"].tolist()
    par_list = para_list[:len(para_list) - 1]
    label_train, label_test, data_train, data_test = train_test_split(label, raw_data[par_list], test_size=0.02)
    eta_list = [0.1]
    # eta_ori =0.1

    F1_best = 0
    F_sc_list = []

    for eta in eta_list:
        m_class = xgb.XGBClassifier(learning_rate=eta, n_estimators=1500, max_depth=7, min_child_weight=6, gamma=0,
                                    subsample=0.8, n_jobs=-1, colsample_bytree=0.8, objective='multi:softmax', num_class=8,
                                    reg_alpha=0.05, reg_lambda=0.05, seed=0)
        # 训练
        m_class.fit(data_train, label_train)
        test_8 = m_class.predict(data_test)
        print("Test Accuracy : %.2f" % accuracy_score(label_test, test_8))
        confusion_mat = confusion_matrix(label_test, test_8)
        print("Test confusion matrix")
        print(confusion_mat)
        F_sc = F1_score(confusion_mat)
        print("test F1_score")
        print(F_sc)
        F_sc_list.append(F_sc)
        if F_sc > F1_best:
            F1_best = F_sc
            best_learning_rate = eta
    print("best F1")
    print(F1_best)
    print("Best learning rate")
    print(best_learning_rate)

    # # plot training and test errors vs number of trees in ensemble
    # plt.plot(eta_list, F_sc_list)
    # plt.xlabel('Attributes')
    # plt.ylabel('F1_score')
    # # plot.ylim([0.0, 1.1*max(mseOob)])
    # plt.show()



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


    ##必须二分类才能计算
    ##print "AUC Score (Train): %f" % metrics.roc_auc_score(y_test, test_2)

    #
    data_submit_raw = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\test_4_combine.csv",
                                   encoding="utf-8", low_memory=False)
    data_submit = data_submit_raw[par_list]
    submit_label_encode = m_class.predict(data_submit)
    decode_list = decode(submit_label_encode)
    user_id_4 = data_submit_raw["user_id"]
    submit_result = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\XGBoost_optimization_totalclass.csv",
                                encoding="utf-8",
                                low_memory=False)
    origin_id = submit_result["user_id"].tolist()
    origin_result = submit_result["current_service"].tolist()
    num_4 = len(user_id_4)
    for i in range(num_4):
        origin_result[origin_id.index(user_id_4[i])] = decode_list[i]
    final_da = pd.DataFrame({"user_id": origin_id, "current_service": origin_result})
    # final_da.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\XGBoost_optimization_final.csv")
