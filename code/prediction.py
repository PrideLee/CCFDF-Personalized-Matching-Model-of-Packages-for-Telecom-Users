import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix


####################    service_type = 4    ###############################


# with open(r"E:\CCFDF\plansmatching\result\class_2_new\rf_combine.pkl", 'rb') as f:
#     model =joblib.load(f)
#
# raw_data = pd.read_csv(
#     r"E:\CCFDF\plansmatching\data\raw data\final_data\test_4_combine.csv",
#      encoding="utf-8", low_memory=False)
#
#
# para_list = ['is_mix_service', 'online_time', '1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee',
#                  '1_total_fee_norm', '2_total_fee_norm', '3_total_fee_norm', '4_total_fee_norm',
#                  'month_traffic', 'many_over_bill', 'contract_type', 'contract_time',
#                  'is_promise_low_consume', 'net_service', 'pay_times', 'pay_num', 'last_month_traffic',
#                  'local_trafffic_month', 'local_caller_time', 'service1_caller_time', 'service2_caller_time', 'gender',
#                  'age', 'complaint_level', 'former_complaint_num', 'former_complaint_fee',
#                  'fee_mean', 'fee_std', 'fee_fluctuate', 'fee_mean_2',
#                  'service_caller_time_fluctuate', 'online_time_norm', 'fee_mean_norm', 'fee_std_norm',
#                  'fee_fluctuate_norm', 'month_traffic_norm', 'contract_time_norm', 'pay_num_norm',
#                  'last_month_traffic_norm', 'local_trafffic_month_norm', 'local_caller_time_norm',
#                  'service1_caller_time_norm', 'service2_caller_time_norm', 'age_norm', 'former_complaint_num_norm',
#                  'former_complaint_fee_norm', 'fee_mean_2_norm', 'service_caller_time_fluctuate_norm']
# data_new = raw_data[para_list]
# prediction = model.predict(data_new)
#
#
#
#
# user_id = raw_data["user_id"].tolist()
# final_re = []
# for i in prediction:
#     if i == 0:
#         final_re.append(89950166)
#     if i == 1:
#         final_re.append(89950167)
#     if i == 2:
#         final_re.append(89950168)
#     if i == 3:
#         final_re.append(99999825)
#     if i == 4:
#         final_re.append(99999826)
#     if i == 5:
#         final_re.append(99999827)
#     if i == 6:
#         final_re.append(99999828)
#     if i == 7:
#         final_re.append(99999830)
#
#
#
# test_data = pd.read_csv(
#     r"E:\CCFDF\plansmatching\data\raw data\final_data\republish_test\republish_test\repiulish_test_duplicate.csv",
#     encoding="utf-8", low_memory=False)
# data_su = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\my_submit.csv", encoding="utf-8",
#                       low_memory=False)
# id_user_test = test_data["user_id"].tolist()
# id_user_submit = data_su["user_id"].tolist()
# result_su = data_su["current_service"].tolist()
# type = test_data["service_type"].tolist()
# num_total = len(type)
# type_1 = [i for i in range(num_total) if type[i] == 1]
# type_1_id = [id_user_test[j] for j in type_1]
# type_3 = [i for i in range(num_total) if type[i] == 3]
# type_3_id = [id_user_test[j] for j in type_3]
# result_1 = [result_su[id_user_submit.index(i)] for i in type_1_id]
# result_3 = [result_su[id_user_submit.index(i)] for i in type_3_id]
# id_remain = user_id + type_1_id + type_3_id
# result_remain = final_re + result_1 + result_3
# resite = [id_remain.index(id_user_submit[i]) for i in range(num_total)]
# re_result = [result_remain[i] for i in resite]
# remain_result = pd.DataFrame({"user_id": id_user_submit, "current_service": re_result})
# remain_result.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\result_combine_2.csv")



####################    service_type = 1    ###############################


# with open(r"E:\CCFDF\plansmatching\result\class_2_new\rf_class_1.pkl", 'rb') as f:
#     model =joblib.load(f)
#
# raw_data = pd.read_csv(
#     r"E:\CCFDF\plansmatching\data\raw data\final_data\class_1_test_new.csv",
#      encoding="utf-8", low_memory=False)
#
#
# para_list = ['is_mix_service', 'online_time', '1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee',
#                  '1_total_fee_norm', '2_total_fee_norm', '3_total_fee_norm', '4_total_fee_norm',
#                  'month_traffic', 'many_over_bill', 'contract_type', 'contract_time',
#                  'is_promise_low_consume', 'net_service', 'pay_times', 'pay_num', 'pay_mean', 'last_month_traffic',
#                  'local_trafffic_month', 'local_caller_time', 'service1_caller_time', 'service2_caller_time', 'gender',
#                  'age', 'complaint_level', 'former_complaint_num', 'former_complaint_fee',
#                  'fee_mean', 'fee_std', 'fee_fluctuate', 'fee_mean_2',
#                  'service_caller_time_fluctuate', 'online_time_norm', 'fee_mean_norm', 'fee_std_norm',
#                  'fee_fluctuate_norm', 'month_traffic_norm', 'contract_time_norm', 'pay_num_norm',
#                  'last_month_traffic_norm', 'local_trafffic_month_norm', 'local_caller_time_norm',
#                  'service1_caller_time_norm', 'service2_caller_time_norm', 'age_norm', 'former_complaint_num_norm',
#                  'former_complaint_fee_norm', 'fee_mean_2_norm', 'service_caller_time_fluctuate_norm',
#                  ]
# data_new = raw_data[para_list]
# prediction = model.predict(data_new)
#
#
# user_id = raw_data["user_id"].tolist()
# final_re = []
# for i in prediction:
#     if i == 8:
#         final_re.append(90063345)
#     if i == 9:
#         final_re.append(90109916)
#     if i == 10:
#         final_re.append(90155946)
#
# remain_result = pd.DataFrame({"user_id": user_id, "current_service": final_re})
# remain_result.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\result_class_1.csv")






# data_su = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\my_submit.csv", encoding="utf-8",
#                       low_memory=False)
# result = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\result_combine_1.csv", encoding="utf-8",
#                       low_memory=False)
# id_user_submit = data_su["user_id"].tolist()
# user_id = result["user_id"].tolist()
# result_remain = result["current_service"].tolist()
# num_total = len(result)
# resite = [user_id.index(id_user_submit[i]) for i in range(num_total)]
# re_result = [result_remain[i] for i in resite]
# remain_result = pd.DataFrame({"user_id": id_user_submit, "current_service": re_result})
# remain_result.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\result_combine_2.csv")



class_total = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\result_combine_2.csv", encoding="utf-8",
                      low_memory=False)
class_1 = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\result_class_1.csv", encoding="utf-8",
                      low_memory=False)
id_total = class_total["user_id"].tolist()
id_1 = class_1["user_id"].tolist()
service_total = class_total["current_service"].tolist()
service_1 = class_1["current_service"].tolist()
num_total = len(id_total)
for i in range(num_total):
    if id_total[i] in id_1:
        service_total[i] = service_1[id_1.index(id_total[i])]

final_da = pd.DataFrame({"user_id": id_total, "current_service": service_total})
final_da.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\result_class_1_4.csv")





