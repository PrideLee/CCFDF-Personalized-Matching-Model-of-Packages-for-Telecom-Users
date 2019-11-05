import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plot
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV

if __name__ == '__main__':

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


    def forest_class(file_dataframe, paralist):
        label = file_dataframe["service_type_encode"].tolist()
        select_data = file_dataframe[paralist]
        attributes_name = np.array(paralist[:len(paralist) - 1])
        label_train, label_test, data_train, data_test = train_test_split(label, select_data, test_size=0.03)

        # select 1/10 data to test, which can  extend to 10 cross-validation
        # nrow = len(label)
        # ixval = 0
        # idxtest = [a for a in range(nrow) if a % 10 == ixval % 10]
        # idxtrain = [a for a in range(nrow) if a % 10 != ixval % 10]
        # label_train = [label[r] for r in idxtrain]
        # label_test = [label[r] for r in idxtest]
        # data_train = select_data.iloc[idxtrain]
        # data_test = select_data.iloc[idxtest]
        id_index_train = data_train["user_id"].tolist()
        id_index_test = data_test["user_id"].tolist()
        data_train_fin = data_train.drop(['user_id'], axis=1)
        data_test_fin = data_test.drop(['user_id'], axis=1)

        # forest train
        missClassError = []
        # random generate 500-5000 trees
        # nTreeList = range(200, 500, 20)

        # sel_num = round(math.log(len(paralist) - 1, 2))
        test_num = len(label_test)
        error_idx = []
        best_correct = 0
        # n_jobs = 4
        iTrees = 470
        # iTrees = 2000
        depth = 36
        maxFeat = 0.36

        # the depth of tree if None the tree will grow continually, until the leaf notes less than min_samples_split
        # depth = 27
        # the attributes num when find the optimum split point, usually select log2(attributes_num)

        RFModel = ensemble.RandomForestClassifier(n_estimators=iTrees, max_depth=depth, max_features=maxFeat, n_jobs=-1,
                                                  oob_score=False, random_state=531)
        RFModel.fit(data_train_fin, label_train)

        # Accumulate auc on test set
        prediction = RFModel.predict(data_test_fin)
        correct = accuracy_score(label_test, prediction)
        prediction_train = RFModel.predict(data_train_fin)
        correct_train = accuracy_score(label_train, prediction_train)
        print("train correct")
        print(correct_train)
        # generate confusion matrix
        pList = prediction.tolist()
        confusionMat = confusion_matrix(label_test, pList)
        print("F1 Score test")
        print(F1_score(confusionMat))

        error_idx += [j for j in range(test_num) if label_test[j] != prediction[j]]
        missClassError.append(1.0 - correct)
        print("test correct")
        print(correct)

        print("Missclassification Error")
        print(missClassError)
        error_id = [id_index_test[m] for m in error_idx]
        # # generate confusion matrix
        # pList = prediction.tolist()
        # confusionMat = confusion_matrix(label_test, pList)

        # Plot feature importance
        featureImportance = RFModel.feature_importances_

        # normalize by max importance
        featureImportance = featureImportance / featureImportance.max()

        # plot variable importance
        idxSorted = np.argsort(featureImportance)
        barPos = np.arange(idxSorted.shape[0]) + .5
        plot.barh(barPos, featureImportance[idxSorted], align='center')
        plot.yticks(barPos, attributes_name[idxSorted])
        plot.xlabel('Variable Importance')
        plot.show()
        # data_temp = data_test
        # data_temp["real_label"] = label_test
        # data_temp.to_csv(r"C:\Users\poder\data_test.csv")
        return error_id, confusionMat, RFModel


    raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\train_combine_4_encode_precentage.csv",
                           encoding="utf-8",
                           low_memory=False)
    # # balance_data = data_balance(raw_data, [0, 1, 6], [0.7, 0.4, 0.2])
    balance_data = raw_data
    # para_list = ['is_mix_service', 'online_time', '1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee',
    #              '1_total_fee_norm', '2_total_fee_norm', '3_total_fee_norm', '4_total_fee_norm',
    #              'month_traffic', 'many_over_bill', 'contract_type', 'contract_time',
    #              'is_promise_low_consume', 'net_service', 'pay_times', 'pay_num', 'last_month_traffic',
    #              'local_trafffic_month', 'local_caller_time', 'service1_caller_time', 'service2_caller_time', 'gender',
    #              'age', 'complaint_level', 'former_complaint_num', 'former_complaint_fee',
    #              'fee_mean', 'fee_std', 'fee_fluctuate', 'fee_mean_2',
    #              'service_caller_time_fluctuate', 'online_time_norm', 'fee_mean_norm', 'fee_std_norm',
    #              'fee_fluctuate_norm', 'month_traffic_norm', 'contract_time_norm', 'pay_num_norm',
    #              'last_month_traffic_norm', 'local_trafffic_month_norm', 'local_caller_time_norm',
    #              'service1_caller_time_norm', 'service2_caller_time_norm', 'age_norm', 'former_complaint_num_norm',
    #              'former_complaint_fee_norm', 'fee_mean_2_norm', 'service_caller_time_fluctuate_norm',
    #              'month_traffic_precentage', 'contract_time_precentage',
    #              'pay_times_precentage', 'pay_num_precentage', 'last_month_traffic_precentage',
    #              'local_trafffic_month_precentage', 'local_caller_time_precentage', 'service1_caller_time_precentage',
    #              'service2_caller_time_precentage', 'month_traffic_precentage', 'contract_time_precentage',
    #              'pay_times_precentage', 'pay_num_precentage', 'last_month_traffic_precentage',
    #              'local_trafffic_month_precentage', 'local_caller_time_precentage', 'service1_caller_time_precentage',
    #              'service2_caller_time_precentage',
    #              'user_id'
    #              ]

    para_list = ['is_mix_service',
                 '1_total_fee_norm', '2_total_fee_norm', '3_total_fee_norm', '4_total_fee_norm',
                 'many_over_bill', 'contract_type',
                 'is_promise_low_consume', 'net_service', 'pay_times',
                 'gender',
                 'complaint_level',
                 'online_time_norm', 'fee_mean_norm', 'fee_std_norm',
                 'fee_fluctuate_norm', 'month_traffic_norm', 'contract_time_norm', 'pay_num_norm',
                 'last_month_traffic_norm', 'local_trafffic_month_norm', 'local_caller_time_norm',
                 'service1_caller_time_norm', 'service2_caller_time_norm', 'age_norm', 'former_complaint_num_norm',
                 'former_complaint_fee_norm', 'fee_mean_2_norm', 'service_caller_time_fluctuate_norm',
                 'month_traffic_precentage', 'contract_time_precentage',
                 'pay_times_precentage', 'pay_num_precentage', 'last_month_traffic_precentage',
                 'local_trafffic_month_precentage', 'local_caller_time_precentage', 'service1_caller_time_precentage',
                 'service2_caller_time_precentage',
                 'user_id'
                 ]

    # para_list = ['is_mix_service', 'online_time', '1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee',
    #              'month_traffic', 'many_over_bill', 'contract_type', 'contract_time',
    #              'is_promise_low_consume', 'net_service', 'pay_times', 'pay_num', 'last_month_traffic',
    #              'local_trafffic_month', 'local_caller_time', 'service1_caller_time', 'service2_caller_time', 'gender',
    #              'age', 'complaint_level', 'former_complaint_num', 'former_complaint_fee',
    #              'fee_mean', 'fee_std', 'fee_fluctuate', 'fee_mean_2',
    #              'service_caller_time_fluctuate', 'month_traffic_precentage', 'contract_time_precentage',
    #              'pay_times_precentage', 'pay_num_precentage', 'last_month_traffic_precentage',
    #              'local_trafffic_month_precentage', 'local_caller_time_precentage', 'service1_caller_time_precentage',
    #              'service2_caller_time_precentage', 'user_id'
    #              ]

    # # id_error, mat_con, rf = forest_class(balance_data, para_list)
    id_error, mat_con, rf = forest_class(balance_data, para_list)
    F1 = F1_score(mat_con)
    print(mat_con)
    print(F1)
    test_data = pd.read_csv(
        r"E:\CCFDF\plansmatching\data\raw data\final_data\test_4_combine.csv",
        encoding="utf-8", low_memory=False)
    para_list_test = para_list[0:len(para_list) - 1]
    data_new = test_data[para_list_test]
    prediction = rf.predict(data_new)
    decode_list = decode(prediction)
    user_id_4 = test_data["user_id"]
    submit_result = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\result_class_1_4.csv",
                                encoding="utf-8",
                                low_memory=False)
    origin_id = submit_result["user_id"].tolist()
    origin_result = submit_result["current_service"].tolist()
    num_4 = len(user_id_4)
    for i in range(num_4):
        origin_result[origin_id.index(user_id_4[i])] = decode_list[i]
    final_da = pd.DataFrame({"user_id": origin_id, "current_service": origin_result})
    final_da.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\result_precentage_norm.csv")

    # joblib.dump(rf, r"E:\CCFDF\plansmatching\result\class_2_new\rf_combine_pre_new.pkl")
    # # joblib.dump(rf, r"C:\Users\poder\rf_test.pkl")
