import pandas as pd
from sklearn.externals import joblib


def decode(decode_type):
    new_type = []
    for i in decode_type:
        if i == 89950166:
            new_type.append(0)
        if i == 89950167:
            new_type.append(1)
        if i == 89950168:
            new_type.append(2)
        if i == 99999825:
            new_type.append(3)
        if i == 99999826:
            new_type.append(4)
        if i == 99999827:
            new_type.append(5)
        if i == 99999828:
            new_type.append(6)
        if i == 99999830:
            new_type.append(7)
        if i == 90063345:
            new_type.append(8)
        if i == 90109916:
            new_type.append(9)
        if i == 90155946:
            new_type.append(10)
    return new_type


def encode(encode_type):
    new_type = []
    for i in encode_type:
        if i == 0:
            new_type.append(89950166)
        if i == 7:
            new_type.append(99999830)
    return new_type


sub = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\result_test\XGBoost_finalprodata.csv",
                  encoding="utf-8", low_memory=False)
sub_type = sub['current_service'].tolist()
id_submit = sub['user_id'].tolist()
num_sub = len(id_submit)
decode_type = decode(sub_type)
class_0_7_id = [id_submit[i] for i in range(num_sub) if (decode_type[i] == 0 or decode_type[i] == 7)]
data_submit_4 = pd.read_csv(
    r"E:\CCFDF\plansmatching\data\raw data\final_data\Chanle_B\test_4_feature_select.csv",
    encoding="utf-8", low_memory=False)
test_data_id = data_submit_4['user_id'].tolist()
num_test = len(test_data_id)
test_0_7_index = []
test_0_7_id = []
for i in range(num_test):
    if test_data_id[i] in class_0_7_id:
        test_0_7_id.append(test_data_id[i])
        test_0_7_index.append(i)
test_0_7_data = data_submit_4.iloc[test_0_7_index]
para_list = ['online_time', '1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee',
             'fee_distance',
             'month_traffic', 'many_over_bill', 'contract_type', 'contract_time', 'roaming_traffic', 'fee_pay_subtract',
             'pay_num', 'last_month_traffic', 'local_trafffic_month', 'local_caller_time', 'service1_caller_time',
             'service2_caller_time', 'age', 'fee_mean', 'fee_std', 'fee_fluctuate', 'fee_mean_2',
             'service_caller_time_fluctuate', 'service_caller_time_mean',
             'fee_mat_subtract_01', 'fee_mat_subtract_12', 'fee_mat_subtract_23', 'fee_mat_sum_01', 'fee_mat_sum_12',
             'fee_mat_sum_23', 'fee_mat_sum_0112', 'pay_mean', 'online_time_per_fee', 'month_traffic_per_fee',
             'contract_time_per_fee', 'last_month_traffic_per_fee', 'local_trafffic_month_per_fee',
             'service_caller_time_mean_per_fee', 'pay_fee_subtract', 'pay_total', 'fee_min', 'traffic_current_month']
type_0_7_se = test_0_7_data[para_list]
model_0_7 = joblib.load(r'E:\CCFDF\plansmatching\data\raw data\final_data\Chanle_B\xgb_0_8_up_real.model')
predict_class_0_7 = model_0_7.predict(type_0_7_se)
predict_class_0_7_encode = encode(predict_class_0_7)
len_0_7 = len(predict_class_0_7)
for i in range(len_0_7):
    sub_type[id_submit.index(test_0_7_id[i])] = predict_class_0_7_encode[i]
pd.DataFrame({'user_id': id_submit, 'current_service': sub_type}).to_csv(
    r'E:\CCFDF\plansmatching\data\raw data\final_data\Chanle_B\XGB_replace_1.csv', index=False)

