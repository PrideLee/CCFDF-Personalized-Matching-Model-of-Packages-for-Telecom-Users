import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plot
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV

if __name__ == '__main__':

    def data_balance(file_dataframe, balance_type, propotion_del):
        type_list = file_dataframe["current_service"].tolist()
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


    def revelant_map(file_dataframe, paras):
        select_dataframe = file_dataframe[paras]
        cormat = pd.DataFrame(select_dataframe.corr())
        plot.pcolor(cormat)
        plot.show()
        return cormat


    def forest_class(file_dataframe, paralist):
        label = file_dataframe["current_service"].tolist()
        select_data = file_dataframe[paralist]
        attributes_name = np.array(paralist[:len(paralist) - 1])
        # label_train, label_test, data_train, data_test = train_test_split(label, select_data, test_size=0.1)

        # select 1/10 data to test, which can  extend to 10 cross-validation
        nrow = len(label)
        ixval = 0
        idxtest = [a for a in range(nrow) if a % 10 == ixval % 10]
        idxtrain = [a for a in range(nrow) if a % 10 != ixval % 10]
        label_train = [label[r] for r in idxtrain]
        label_test = [label[r] for r in idxtest]
        data_train = select_data.iloc[idxtrain]
        data_test = select_data.iloc[idxtest]
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
        iTrees = 480
        depth_list = 36
        for depth in depth_list:
            # the depth of tree if None the tree will grow continually, until the leaf notes less than min_samples_split
            # depth = 27
            # the attributes num when find the optimum split point, usually select log2(attributes_num)
            maxFeat = 0.4
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
            if correct > best_correct:
                best_correct = correct
                best_rf = RFModel
                best_confusionnMat = confusionMat
                error_idx_best = [j for j in range(test_num) if label_test[j] != prediction[j]]
            error_idx += [j for j in range(test_num) if label_test[j] != prediction[j]]
            missClassError.append(1.0 - correct)
            print("test correct")
            print(correct)
        print("Best correct")
        print(best_correct)
        print("Missclassification Error")
        print(missClassError)
        error_id = [id_index_test[m] for m in error_idx]
        error_id_best = [id_index_test[n] for n in error_idx_best]
        # # generate confusion matrix
        # pList = prediction.tolist()
        # confusionMat = confusion_matrix(label_test, pList)
        print('')
        print("Best Confusion Matrix")
        print(best_confusionnMat)

        # plot training and test errors vs number of trees in ensemble
        plot.plot(depth_list, missClassError)
        plot.xlabel('Number of Trees in Ensemble')
        plot.ylabel('Missclassification Error Rate')
        # plot.ylim([0.0, 1.1*max(mseOob)])
        plot.show()

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
        return error_id, error_id_best, best_confusionnMat, best_rf


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


    # def forest_class(file_dataframe, paralist):
    #     label = file_dataframe["current_service"].tolist()
    #     select_data = file_dataframe[paralist]
    #     attributes_name = np.array(paralist[:len(paralist) - 1])
    #     # label_train, label_test, data_train, data_test = train_test_split(label, select_data, test_size=0.1)
    #
    #     # select 1/10 data to test, which can  extend to 10 cross-validation
    #     nrow = len(label)
    #     ixval = 0
    #     idxtest = [a for a in range(nrow) if a % 10 == ixval % 10]
    #     idxtrain = [a for a in range(nrow) if a % 10 != ixval % 10]
    #     label_train = [label[r] for r in idxtrain]
    #     label_test = [label[r] for r in idxtest]
    #     data_train = select_data.iloc[idxtrain]
    #     data_test = select_data.iloc[idxtest]
    #     id_index_train = data_train["user_id"].tolist()
    #     id_index_test = data_test["user_id"].tolist()
    #     data_train_fin = data_train.drop(['user_id'], axis=1)
    #     data_test_fin = data_test.drop(['user_id'], axis=1)
    #
    #     # forest train
    #     missClassError = []
    #     # random generate 500-5000 trees
    #     # nTreeList = range(250, 350, 10)
    #
    #
    #     sel_num = round(math.log(len(paralist) - 1, 2))
    #     test_num = len(label_test)
    #     error_idx = []
    #
    #
    #
    #     iTrees = 300
    #     best_correct = 0
    #     ndeepth = range(27, 50)
    #     for depth in ndeepth:
    #         # the attributes num when find the optimum split point, usually select log2(attributes_num)
    #         maxFeat = sel_num
    #         RFModel = ensemble.RandomForestClassifier(n_estimators=iTrees, max_depth=depth, max_features=maxFeat,
    #                                                   oob_score=False, random_state=531)
    #         RFModel.fit(data_train_fin, label_train)
    #
    #         # Accumulate auc on test set
    #         prediction = RFModel.predict(data_test_fin)
    #         correct = accuracy_score(label_test, prediction)
    #         prediction_train = RFModel.predict(data_train_fin)
    #         correct_train = accuracy_score(label_train, prediction_train)
    #         print("train correct")
    #         print(correct_train)
    #         # generate confusion matrix
    #         pList = prediction.tolist()
    #         confusionMat = confusion_matrix(label_test, pList)
    #         if correct > best_correct:
    #             best_correct = correct
    #             best_rf = RFModel
    #             best_confusionnMat = confusionMat
    #             error_idx_best = [j for j in range(test_num) if label_test[j] != prediction[j]]
    #         error_idx += [j for j in range(test_num) if label_test[j] != prediction[j]]
    #         missClassError.append(1.0 - correct)
    #         print("test correct")
    #         print(correct)
    #         print("F1_score")
    #         print(F1_score(confusionMat))
    #     print("Best correct")
    #     print(best_correct)
    #     print("Missclassification Error")
    #     print(missClassError)
    #     error_id = [id_index_test[m] for m in error_idx]
    #     error_id_best = [id_index_test[n] for n in error_idx_best]
    #     # # generate confusion matrix
    #     # pList = prediction.tolist()
    #     # confusionMat = confusion_matrix(label_test, pList)
    #     print('')
    #     print("Best Confusion Matrix")
    #     print(best_confusionnMat)
    #
    #     # plot training and test errors vs number of trees in ensemble
    #     plot.plot(ndeepth, missClassError)
    #     plot.xlabel('Number of Trees deep in Ensemble')
    #     plot.ylabel('Missclassification Error Rate')
    #     # plot.ylim([0.0, 1.1*max(mseOob)])
    #     plot.show()
    #
    #     # Plot feature importance
    #     featureImportance = RFModel.feature_importances_
    #
    #     # normalize by max importance
    #     featureImportance = featureImportance / featureImportance.max()
    #
    #     # plot variable importance
    #     idxSorted = np.argsort(featureImportance)
    #     barPos = np.arange(idxSorted.shape[0]) + .5
    #     plot.barh(barPos, featureImportance[idxSorted], align='center')
    #     plot.yticks(barPos, attributes_name[idxSorted])
    #     plot.xlabel('Variable Importance')
    #     plot.show()
    #     return error_id, error_id_best, best_confusionnMat, best_rf

    # raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_proprocess.csv", encoding="utf-8",
    #                        low_memory=False)
    # para_list = ['is_mix_service', 'online_time_norm', '1_total_fee', '2_total_fee',
    #              '3_total_fee', '4_total_fee', 'month_traffic_norm', 'many_over_bill', 'contract_type',
    #              'contract_time_norm', 'is_promise_low_consume', 'net_service', 'pay_times', 'pay_num_norm',
    #              'last_month_traffic_norm', 'local_trafffic_month_norm', 'local_caller_time_norm',
    #              'service1_caller_time_norm', 'service2_caller_time_norm', 'gender', 'age_norm', 'complaint_level',
    #              'former_complaint_num_norm', 'former_complaint_fee_norm', 'fee_mean', 'fee_std', 'fee_fluctuate',
    #              'service1_caller_time_dimean', 'last_month_traffic_dimean', 'former_complaint_num_dimean',
    #              'former_complaint_fee_dimean', 'current_service']
    # revelant_value = revelant_map(raw_data, para_list)
    # revelant_value.to_csv(r"E:\CCFDF\plansmatching\data\data analysis _2\revelant.csv", encoding="utf-8")
    # dataframe_balance = data_balance(raw_data, [4, 5, 7], [0.5, 0.2, 0.72])
    # dataframe_balance.to_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_balance.csv", encoding="utf-8")

    # raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_sup.csv", encoding="utf-8",
    #                        low_memory=False)
    # print(raw_data["current_service"].value_counts())
    # dataframe_balance = data_balance(raw_data, [4, 5, 14, 1, 6, 13, 12, 0, 11], [0.875, 0.75, 0.7, 0.7, 0.5, 0.5, 0.5, 0.25, 0.25])
    # dataframe_balance.to_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_sup_balance.csv", encoding="utf-8")

    # def forest_class(file_dataframe, paralist):
    #     label = file_dataframe["current_service"].tolist()
    #     select_data = file_dataframe[paralist]
    #     attributes_name = np.array(paralist[:len(paralist) - 1])
    #     label_train, label_test, data_train, data_test = train_test_split(label, select_data, test_size=0.1)
    #
    #     # select 1/10 data to test, which can  extend to 10 cross-validation
    #     # nrow = len(label)
    #     # ixval = 0
    #     # idxtest = [a for a in range(nrow) if a % 10 == ixval % 10]
    #     # idxtrain = [a for a in range(nrow) if a % 10 != ixval % 10]
    #     # label_train = [label[r] for r in idxtrain]
    #     # label_test = [label[r] for r in idxtest]
    #     # data_train = select_data.iloc[idxtrain]
    #     # data_test = select_data.iloc[idxtest]
    #     id_index_train = data_train["user_id"].tolist()
    #     id_index_test = data_test["user_id"].tolist()
    #     data_train_fin = data_train.drop(['user_id'], axis=1)
    #     data_test_fin = data_test.drop(['user_id'], axis=1)
    #
    #     # forest train
    #     missClassError = []
    #     # random generate 500-5000 trees
    #     # nTreeList = range(200, 400, 10)
    #
    #     # sel_num = round(math.log(len(paralist) - 1, 2))
    #     test_num = len(label_test)
    #     error_idx = []
    #     best_correct = 0
    #     iTrees = 350
    #     maxFeat_list = [0.4, 0.6]
    #     # min_sample_split_pre = 0.08
    #     for maxFeat in maxFeat_list:
    #         # the depth of tree if None the tree will grow continually, until the leaf notes less than min_samples_split
    #         depth = 27
    #         # the attributes num when find the optimum split point, usually select log2(attributes_num)
    #         # maxFeat = "log2"
    #         RFModel = ensemble.RandomForestClassifier(n_estimators=iTrees, max_depth=depth,
    #                                                   n_jobs=-1,
    #                                                   max_features=maxFeat, oob_score=False, random_state=531)
    #         RFModel.fit(data_train_fin, label_train)
    #
    #         # Accumulate auc on test set
    #         prediction = RFModel.predict(data_test_fin)
    #         correct = accuracy_score(label_test, prediction)
    #         prediction_train = RFModel.predict(data_train_fin)
    #         correct_train = accuracy_score(label_train, prediction_train)
    #         print("train correct")
    #         print(correct_train)
    #         # generate confusion matrix
    #         pList = prediction.tolist()
    #         confusionMat = confusion_matrix(label_test, pList)
    #         print("F1 Score test")
    #         print(F1_score(confusionMat))
    #         if correct > best_correct:
    #             best_correct = correct
    #             best_rf = RFModel
    #             best_confusionnMat = confusionMat
    #             error_idx_best = [j for j in range(test_num) if label_test[j] != prediction[j]]
    #         error_idx += [j for j in range(test_num) if label_test[j] != prediction[j]]
    #         missClassError.append(1.0 - correct)
    #         print("test correct")
    #         print(correct)
    #     print("Best correct")
    #     print(best_correct)
    #     print("Missclassification Error")
    #     print(missClassError)
    #     error_id = [id_index_test[m] for m in error_idx]
    #     error_id_best = [id_index_test[n] for n in error_idx_best]
    #     # # generate confusion matrix
    #     # pList = prediction.tolist()
    #     # confusionMat = confusion_matrix(label_test, pList)
    #     print('')
    #     print("Best Confusion Matrix")
    #     print(best_confusionnMat)
    #
    #     # plot training and test errors vs number of trees in ensemble
    #     plot.plot(maxFeat_list, missClassError)
    #     plot.xlabel('Number of Trees in Ensemble')
    #     plot.ylabel('Missclassification Error Rate')
    #     # plot.ylim([0.0, 1.1*max(mseOob)])
    #     plot.show()
    #
    #     # Plot feature importance
    #     featureImportance = RFModel.feature_importances_
    #
    #     # normalize by max importance
    #     featureImportance = featureImportance / featureImportance.max()
    #
    #     # plot variable importance
    #     idxSorted = np.argsort(featureImportance)
    #     barPos = np.arange(idxSorted.shape[0]) + .5
    #     plot.barh(barPos, featureImportance[idxSorted], align='center')
    #     plot.yticks(barPos, attributes_name[idxSorted])
    #     plot.xlabel('Variable Importance')
    #     plot.show()
    #     return error_id, error_id_best, best_confusionnMat, best_rf

    def forest_class_grid(file_dataframe, paralist):
        label = file_dataframe["current_service_new"].tolist()
        select_data = file_dataframe[paralist]
        attributes_name = np.array(paralist[:len(paralist) - 1])
        label_train, label_test, data_train, data_test = train_test_split(label, select_data, test_size=0.3)

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
        # nTreeList = range(200, 400, 10)

        # sel_num = round(math.log(len(paralist) - 1, 2))
        test_num = len(label_test)
        error_idx = []
        iTrees = 350
        # maxFeat = 0.4
        # the depth of tree if None the tree will grow continually, until the leaf notes less than min_samples_split
        # depth = 27
        # the attributes num when find the optimum split point, usually select log2(attributes_num)
        # maxFeat = "log2"
        estimator = ensemble.RandomForestClassifier(n_estimators=iTrees)
        parameters = {'max_depth': range(15, 40), "max_features": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}
        grid = RandomizedSearchCV(estimator=estimator, param_distributions=parameters, n_iter=10, verbose=1,
                                  random_state=77, n_jobs=-1, cv=5)
        grid.fit(data_train_fin, label_train)
        print('网格搜索-度量记录：', grid.cv_results_)  # 包含每次训练的相关信息
        print('网格搜索-最佳度量值:', grid.best_score_)  # 获取最佳度量值
        print('网格搜索-最佳参数：', grid.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典
        print('网格搜索-最佳模型：', grid.best_estimator_)  # 获取最佳度量时的分类器模型
        RFModel = ensemble.RandomForestClassifier(n_estimators=iTrees, max_depth=grid.best_params_["max_depth"],
                                                  max_features=grid.best_params_["max_features"], n_jobs=-1,
                                                  oob_score=False, random_state=77)

        # RFModel = ensemble.RandomForestClassifier(n_estimators=iTrees, max_depth=depth,
        #                                           n_jobs=4, max_features=maxFeat, oob_score=False, random_state=531)
        # RFModel.fit(data_train_fin, label_train)

        # Accumulate auc on test set
        prediction = RFModel.predict(data_test_fin)
        # prediction_pro = RFModel.predict_proba(data_test_fin)
        # num = len(prediction)
        # for i in range(num):
        #     if (prediction[i] != label_test[i]) & (label_test[i] != 0):
        #         print(prediction_pro[i])
        #         print(label_test[i])
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
        print('')
        print("Confusion Matrix")
        print(confusionMat)

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
        return error_id, confusionMat, RFModel


    # raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_sup_balance.csv", encoding="utf-8",
    #                        low_memory=False)

    # raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_sup_add_0_correct.csv", encoding="utf-8",
    # low_memory = False)
    # dataframe_balance = data_balance(raw_data, [4, 5, 1, 14], [0.42, 0.25, 0.2, 0.2])
    # dataframe_balance.to_csv(r"E:\CCFDF\plansmatching\data\raw data\train\balance_1_correct.csv")

    dataframe_balance = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\train\balance_1_correct.csv",
                                    encoding="utf-8", low_memory=False)
    select_par = ['online_time', 'month_traffic', 'contract_type', 'contract_time', 'pay_num',
                  'last_month_traffic', 'local_trafffic_month', 'local_caller_time',
                  'service1_caller_time', 'service2_caller_time', 'age_norm',
                  'fee_mean', 'fee_std', 'fee_mean_2_norm', 'service_caller_time_fluctuate',
                  'fee_fluctuate', 'user_id']
    id_error, best_id_error, confusion_matrix_best, rf_best = forest_class(dataframe_balance, select_par)
    F1_sco = F1_score(confusion_matrix_best)
    print("Best_F1_score")
    print(F1_sco)
    id_error_dataframe = pd.DataFrame({"error_id": id_error})
    best_id_error_dataframe = pd.DataFrame({"best_error_id": best_id_error})
    # id_error_dataframe.to_csv(r"E:\CCFDF\plansmatching\result\class_2\RF\error_id_balance_correct_0.csv",
    #                           encoding="utf-8")
    # best_id_error_dataframe.to_csv(r"E:\CCFDF\plansmatching\result\class_2\RF\best_error_id_balance_correct_0.csv",
    #                                encoding="utf-8")
    # joblib.dump(rf_best, r"E:\CCFDF\plansmatching\result\class_2\RF\best_rf_5_balance_correct_0.pkl")

    # raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_sup_add_0.csv", encoding="utf-8",
    #                        low_memory=False)
    # dataframe_balance = data_balance(raw_data, [4, 5, 1, 14], [0.42, 0.25, 0.2, 0.2])
    # dataframe_balance.to_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_sup_add_0_balance.csv")
    # dataframe_balance = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_sup_add_0_balance.csv", encoding="utf-8",
    #                                  low_memory=False)
    # select_par = ['online_time_norm', 'month_traffic_norm', 'contract_type', 'contract_time_norm', 'pay_num_norm',
    #               'last_month_traffic_norm', 'local_trafffic_month_norm', 'local_caller_time_norm',
    #               'service1_caller_time_norm', 'service2_caller_time_norm', 'age_norm',
    #               'fee_mean', 'fee_mean_2', 'fee_std', 'fee_mean_2_norm',
    #               'fee_fluctuate', 'month_traffic_hierarchy', 'contract_time_hierarchy', 'pay_nums_hierarchy',
    #               'last_month_traffic_hierarchy', 'local_trafffic_month_hierarchy', 'local_caller_time_hierarchy',
    #               'service1_caller_time_hierarchy',
    #               'user_id']

    # raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\train\small_class_12_13_14_others.csv",
    #                        encoding="utf-8", low_memory=False)
    # dataframe_balance = data_balance(raw_data, [0, 14], [0.5, 0.3])
    # select_par = ['online_time_norm', 'contract_type', 'pay_num_norm',
    #               'local_trafffic_month_norm',
    #               'service2_caller_time_norm', 'age_norm',
    #               'fee_mean', 'fee_mean_2', 'fee_std', 'fee_mean_2_norm',
    #               'fee_fluctuate', 'month_traffic_norm', 'contract_time_norm', 'pay_nums_hierarchy',
    #               'last_month_traffic_norm', 'local_trafffic_month_norm', 'local_caller_time_norm',
    #               'service1_caller_time_norm',
    #               'user_id']
    #
    # id_error, confusion_matrix_best, rf_best = forest_class(dataframe_balance, select_par)
    # # id_error, best_id_error, confusion_matrix_best, rf_best = forest_class(dataframe_balance, select_par)
    # F1_sco = F1_score(confusion_matrix_best)
    # print("Best_F1_score")
    # print(F1_sco)
    # id_error_dataframe = pd.DataFrame({"error_id": id_error})
    # best_id_error_dataframe = pd.DataFrame({"best_error_id": best_id_error})
    # id_error_dataframe.to_csv(r"E:\CCFDF\plansmatching\result\class_2\RF\error_id_5_balance_1.csv", encoding="utf-8")
    # best_id_error_dataframe.to_csv(r"E:\CCFDF\plansmatching\result\class_2\RF\best_error_id_5_balance_1.csv", encoding="utf-8")
    # joblib.dump(rf_best, r"E:\CCFDF\plansmatching\result\class_2\RF\best_rf_5_balance_1.pkl")
