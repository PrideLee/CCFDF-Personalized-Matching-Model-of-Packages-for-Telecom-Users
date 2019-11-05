import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plot
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
from sklearn import ensemble


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



raw_data = pd.read_csv(r"/Users/peterlee/Documents/CCFDF18/final_data/train_combine_4_encode_precentage.csv",
                       encoding="utf-8", low_memory=False)
raw_data = raw_data.iloc[0:100]
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
             'user_id'
             ]
attributes_name = np.array(para_list[:len(para_list) - 1])
id_list = raw_data["user_id"].tolist()
num_total = len(raw_data)
ixval = 0
idxtest = [a for a in range(num_total) if a % 10 == ixval % 10]
idxtrain = [a for a in range(num_total) if a % 10 != ixval % 10]
label = raw_data["service_type_encode"].tolist()
raw_data = raw_data[para_list]

# test_id = [id_list[i] for i in idxtest]
# test_df = pd.DataFrame({"user_id":test_id})
# test_df.to_csv(r"/Users/peterlee/Documents/CCFDF18/final_data/test_id.csv")
# train_id = [id_list[i] for i in idxtrain]
# train_df = pd.DataFrame({"user_id":train_id})
# train_df.to_csv(r"/Users/peterlee/Documents/CCFDF18/final_data/train_id.csv")
label_train = [label[r] for r in idxtrain]
label_test = [label[r] for r in idxtest]
data_train = raw_data.iloc[idxtrain]
data_test = raw_data.iloc[idxtest]
data_train_fin = data_train.drop(['user_id'], axis=1)
data_test_fin = data_test.drop(['user_id'], axis=1)
missClassError = []
iTrees = 470
depth = 36
maxFeat = 0.36
classweight = None


RFModel = ensemble.RandomForestClassifier(n_estimators=iTrees, max_depth=depth, max_features=maxFeat, n_jobs=-1,
                                          class_weight=classweight, oob_score=True, random_state=531)
RFModel.fit(data_train_fin, label_train)
print(RFModel.predict_proba(data_train_fin))
print("obb_score")
print(RFModel.oob_score_)
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

missClassError.append(1.0 - correct)
print("test correct")
print(correct)

print("Missclassification Error")
print(missClassError)


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
print("importance attributes")
print(attributes_name[idxSorted])
print("importance score")
print(featureImportance[idxSorted])


print(prediction_train)










