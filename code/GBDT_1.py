from math import sqrt, fabs, exp
import matplotlib.pyplot as plot
from sklearn.linear_model import enet_path
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn import ensemble
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.externals import joblib


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
    print("F1 Score")
    print(F1)
    print("Precision")
    print(precision)
    print("Recall")
    print(recall)
    F_score = ((1 / len(F1)) * sum(F1)) ** 2
    return F_score


def GBDT(file_dataframe, paralist):
    # arrange data into list for labels and list of lists for attributes
    label = file_dataframe["current_sample_reencode"].tolist()
    select_data = file_dataframe[paralist]
    label_train, label_test, data_train, data_test = train_test_split(label, select_data, test_size=0.1)

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

    # instantiate model
    nEst = 400
    depth = 27
    learnRate = 0.005
    maxFeatures = "log2"
    subSamp = 0.6
    GBMModel = ensemble.GradientBoostingClassifier(n_estimators=nEst, max_depth=depth,
                                                   learning_rate=learnRate, max_features=maxFeatures,
                                                   subsample=subSamp)
    # train
    GBMModel.fit(data_train_fin, label_train)

    # compute auc on test set as function of ensemble size
    missClassError = []
    missClassBest = 1.0
    predictions = GBMModel.staged_decision_function(data_test_fin)
    # prediction is a three dimension matrix: nEst * each prediction matrix.
    for p in predictions:
        missClass = 0
        for j in range(len(p)):
            listP = p[j].tolist()

            if listP.index(max(listP)) != label_test[j]:
                missClass += 1
        missClass = float(missClass) / len(p)

        missClassError.append(missClass)

        # capture best predictions
        if missClass < missClassBest:
            missClassBest = missClass
            pBest = p
    print("missclassError")
    print(missClassError)
    idxBest = missClassError.index(min(missClassError))

    # print best values
    print("Best Missclassification Error")
    print(missClassBest)
    print("Number of Trees for Best Missclassification Error")
    print(idxBest)

    # plot training deviance and test auc's vs number of trees in ensemble
    num_total = len(data_test_fin)
    missClassError = [mce * num_total for mce in missClassError]
    print('Training Set Deviance')
    print(GBMModel.train_score_)
    print('Test Set Error')
    print(missClassError)
    plot.figure()
    plot.plot(range(1, nEst + 1), GBMModel.train_score_, label='Training Set Deviance', linestyle=":")
    plot.plot(range(1, nEst + 1), missClassError, label='Test Set Error')
    plot.legend(loc='upper right')
    plot.xlabel('Number of Trees in Ensemble')
    plot.ylabel('Deviance / Classification Error')
    plot.show()

    # Plot feature importance
    featureImportance = GBMModel.feature_importances_

    # normalize by max importance
    featureImportance = featureImportance / featureImportance.max()

    # plot variable importance
    idxSorted = np.argsort(featureImportance)
    attributes_name = np.array(paralist[:len(paralist) - 1])
    print('Variable Importance')
    print(attributes_name[idxSorted])
    print(featureImportance[idxSorted])
    barPos = np.arange(idxSorted.shape[0]) + .5
    plot.barh(barPos, featureImportance[idxSorted], align='center')
    plot.yticks(barPos, attributes_name[idxSorted])
    plot.xlabel('Variable Importance')
    plot.show()

    # generate confusion matrix for best prediction.
    pBestList = pBest.tolist()
    bestPrediction = [r.index(max(r)) for r in pBestList]
    num = len(bestPrediction)
    error = []
    for i in range(num):
        if bestPrediction[i] != label_test:
            error.append(i)
    id_error = [id_index_test[j] for j in error]
    confusionMat = confusion_matrix(label_test, bestPrediction)
    print('')
    print("Confusion Matrix")
    print(confusionMat)
    return confusionMat, id_error, GBMModel, GBMModel.train_score_, missClassError, attributes_name[idxSorted], \
           featureImportance[idxSorted]


select_par = ['online_time_norm', 'contract_type', 'pay_num_norm', 'is_mix_service', 'many_over_bill', 'net_service',
              'is_promise_low_consume', 'pay_times', 'gender', 'complaint_level', 'former_complaint_num',
              'local_trafffic_month_norm', 'service_caller_time_fluctuate_norm',
              'service2_caller_time_norm', 'age_norm',
              'fee_mean', 'fee_std', 'fee_mean_2_norm',
              'fee_fluctuate', 'month_traffic_norm', 'contract_time_norm', 'pay_nums_hierarchy',
              'last_month_traffic_norm', 'local_trafffic_month_norm', 'local_caller_time_norm',
              'service1_caller_time_norm',
              'user_id']

dataframe_balance = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\train\class_2_sup_add_0_balance_sample.csv",
                                encoding="utf-8",
                                low_memory=False)

confusion_mat, error_id, BestModel, train_score, class_error, importace_attributes, attributes_score = GBDT(
    dataframe_balance, select_par)
print("F1_score")
F1_score(confusion_mat)
id_error_dataframe = pd.DataFrame({"error_id": error_id})
id_error_dataframe.to_csv(r"E:\CCFDF\plansmatching\result\class_2\GBDT\GBDT_error_id_0.csv", encoding="utf-8")
train_score_dataframe = pd.DataFrame({"train_score": train_score})
train_score_dataframe.to_csv(r"E:\CCFDF\plansmatching\result\class_2\GBDT\GBDT_train_score_0.csv", encoding="utf-8")
class_error_dataframe = pd.DataFrame({"class_error": class_error})
class_error_dataframe.to_csv(r"E:\CCFDF\plansmatching\result\class_2\GBDT\GBDT_class_error_0.csv", encoding="utf-8")
importace_attributes_dataframe = pd.DataFrame({"importace_attributes": importace_attributes})
importace_attributes_dataframe.to_csv(r"E:\CCFDF\plansmatching\result\class_2\GBDT\GBDT_importace_attributes_0.csv",
                                      encoding="utf-8")
attributes_score_dataframe = pd.DataFrame({"attributes_score": attributes_score})
attributes_score_dataframe.to_csv(r"E:\CCFDF\plansmatching\result\class_2\GBDT\GBDT_attributes_score_0.csv",
                                  encoding="utf-8")
joblib.dump(BestModel, r"E:\CCFDF\plansmatching\result\class_2\GBDT\best_GBDT_0.pkl")