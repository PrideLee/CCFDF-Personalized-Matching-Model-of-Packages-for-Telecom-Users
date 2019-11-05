import pandas as pd
import numpy as np


def roaming_traffic(dataframe):
    temp = dataframe['month_traffic'] - dataframe['local_trafffic_month']
    dataframe['roaming_traffic'] = temp.tolist()
    return dataframe


def fee_pay_subtract(dataframe):
    temp = dataframe['1_total_fee'] + dataframe['2_total_fee'] + dataframe['3_total_fee'] + dataframe['4_total_fee'] \
           - dataframe['pay_total']
    dataframe['fee_pay_subtract'] = temp.tolist()
    return dataframe


def age_hierarchy(dataframe):
    temp = dataframe['age'].tolist()
    age_hierarchy = []
    for i in temp:
        if 16 <= i < 26:
            age_hierarchy.append(1)
        else:
            age_hierarchy.append(0)
    dataframe['age_hierarchy'] = age_hierarchy
    return dataframe


def feature(dataframe):
    fea_1 = roaming_traffic(dataframe)
    fea_2 = fee_pay_subtract(fea_1)
    fea_3 = age_hierarchy(fea_2)
    return fea_3


train_1 = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\train1_final.csv",
                       encoding="utf-8", low_memory=False)
train_4 = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\train4_feature_2.csv",
                       encoding="utf-8", low_memory=False)
test_1 = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\test1_final.csv",
                       encoding="utf-8", low_memory=False)
test_4 = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\test4_feature_2.csv",
                       encoding="utf-8", low_memory=False)
train_1_fea = feature(train_1)
train_4_fea = feature(train_4)
test_1_fea = feature(test_1)
test_4_fea = feature(test_4)

# feature_select = ['service_type', 'fee_min', 'traffic_current_month', 'contract_time', 'pay_total', 'local_caller_time',
#                   'roaming_traffic', 'online_time', 'contract_type', 'fee_pay_subtract', 'service_caller_time_mean',
#                   'age', 'age_hierarchy']
# train_1_feature_select = train_1_fea[feature_select]
# train_4_feature_select = train_1_fea[feature_select]
# test_1_feature_select = test_1_fea[feature_select]
# test_4_feature_select = test_4_fea[feature_select]
train_1_fea.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\Chanle_B\train_1_feature_select.csv",
                              index=False)
train_4_fea.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\Chanle_B\train_4_feature_select.csv",
                              index=False)
test_1_fea.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\Chanle_B\test_1_feature_select.csv",
                              index=False)
test_4_fea.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\Chanle_B\test_4_feature_select.csv",
                              index=False)





