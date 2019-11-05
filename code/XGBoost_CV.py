import pandas as pd
import numpy as np
import xgboost as xgb
import random
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import warnings

if __name__ == '__main__':

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


    def XGBoost_randomcv(X_train, y_train, X_test, test_label, paras_list):
        # cv_params = {'n_estimators': [500, 800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000]}
        # other_params = {'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
        #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1,
        #                 'n_jobs': -1,
        #                 'objective': 'binary:logistic'}

        # cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8]}
        # other_params = {'learning_rate': 0.1, 'n_estimators': 1500, 'seed': 0,
        #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0.05, 'reg_lambda': 0.05,
        #                 'objective': 'binary:logistic'}
        other_params = {'learning_rate': 0.1, 'max_depth': 7, 'min_child_weight': 6, 'seed': 0,
                        'gamma': 0, 'reg_alpha': 0, 'n_estimators': 1500,
                        'reg_lambda': 1, 'n_jobs': -1, 'objective': 'binary:logistic'}
        cv_params = {'subsample': [0.2, 0.4, 0.5, 0.6, 0.8], 'colsample_bytree': [0.2, 0.4, 0.5, 0.6, 0.8]}
        model = xgb.XGBClassifier(**other_params)
        optimized_GBM = RandomizedSearchCV(estimator=model, param_distributions=cv_params, scoring='r2', cv=3,
                                           n_jobs=-1, n_iter=8)
        optimized_GBM.fit(X_train, y_train)
        evalute_result = optimized_GBM.cv_results_
        print('每轮迭代运行结果:{0}'.format(evalute_result))
        print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
        print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
        # best_model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=optimized_GBM.best_params_["n_estimators"],
        #                                max_depth=5, min_child_weight=1, seed=0,
        #                                subsample=0.8, colsample_bytree=0.8, gamma=0, reg_alpha=0, reg_lambda=1, n_jobs=-1,
        #                                ojective='binary:logistic' )
        #
        # best_model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=1500,
        #                                max_depth=optimized_GBM.best_params_["max_depth"],
        #                                min_child_weight=optimized_GBM.best_params_["min_child_weight"], seed=0,
        #                                subsample=0.8, colsample_bytree=0.8, gamma=0, reg_alpha=0, reg_lambda=1,
        #                                n_jobs=-1,
        #                                objective='binary:logistic')

        best_model = xgb.XGBClassifier(learning_rate=0.1, n_estimators=1500, max_depth=7, min_child_weight=6, seed=0,
                                       subsample=optimized_GBM.best_params_["subsample"],
                                       colsample_bytree=optimized_GBM.best_params_["colsample_bytree"], gamma=0,
                                       reg_alpha=0.05, reg_lambda=0.05, n_jobs=-1, objective='binary:logistic')
        best_model.fit(X_train, y_train)
        y_test = best_model.predict(X_test)
        print("Accuracy : %.2f" % accuracy_score(test_label, y_test))
        confusion_mat = confusion_matrix(test_label, y_test)
        print("Test confusion matrix")
        print(confusion_mat)
        F_sc = F1_score(confusion_mat)
        print("Best model test F1_score")
        print(F_sc)

        # 查看重要程度
        attributes_name = np.array(paras_list[:len(paras_list)])
        featureImportance = best_model.feature_importances_
        idxSorted = np.argsort(featureImportance)
        barPos = np.arange(idxSorted.shape[0]) + .5
        plt.barh(barPos, featureImportance[idxSorted], align='center')
        plt.yticks(barPos, attributes_name[idxSorted])
        plt.xlabel('Variable Importance')
        plt.show()
        return best_model, optimized_GBM.best_params_


    raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\train_combine_4_encode_precentage.csv",
                           encoding="utf-8", low_memory=False)
    # raw_data = pd.read_csv(r"/Users/peterlee/Documents/CCFDF18/final_data/train_combine_4_encode_precentage.csv",
    #                        encoding="utf-8",
    #                        low_memory=False)
    num_total = len(raw_data)
    random_site = random.sample(range(num_total), round(num_total * 0.2))

    raw_data = raw_data.iloc[random_site]

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
                 'service2_caller_time_precentage',
                 'user_id']

    label = raw_data["service_type_encode"].tolist()
    par_list = para_list[:len(para_list) - 1]
    label_train, label_test, data_train, data_test = train_test_split(label, raw_data[par_list], test_size=0.1)
    best_classfier, best_attributes = XGBoost_randomcv(data_train, label_train, data_test, label_test, par_list)
    print(best_attributes)
