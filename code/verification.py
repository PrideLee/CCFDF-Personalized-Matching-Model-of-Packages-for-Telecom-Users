import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix


with open(r"E:\CCFDF\plansmatching\result\class_2_new\rf_best_0.pkl", 'rb') as f:
    model =joblib.load(f)

raw_data = pd.read_csv(
    r"E:\CCFDF\plansmatching\data\raw data\final_data\class_2_pro_1.csv",
    encoding="utf-8", low_memory=False)
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
                 'former_complaint_fee_norm', 'fee_mean_2_norm', 'service_caller_time_fluctuate_norm']
data_new = raw_data[para_list]
prediction = model.predict(data_new)
user_id = raw_data["user_id"].tolist()
final_re = []
for i in prediction:
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




# test_file = 'valid.csv'
#
# predict_file = 'my_test.csv'


# def read_data(filename):
#     data = []
#     with open(filename) as csvfile:
#         csv_reader = csv.reader(csvfile)
#         header = next(csv_reader)
#         for row in csv_reader:
#             if row[1] not in SERVICE:
#                 raise ValueError("Check you label name. "
#                                  "%s is an error uid:" % row[1])
#             data.append(row[1])
#     return data


# y_pred = read_data(predict_file)  # read predict
# y_true = read_data(test_file)  # read test

# num = len(real)
# same = []
# for i in range(num):
#     if real[i] == final_re[i]:
#         same.append(1)
# print(sum(same)/num)

