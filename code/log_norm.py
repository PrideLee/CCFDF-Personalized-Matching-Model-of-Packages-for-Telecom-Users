import pandas as pd
import numpy as np
from math import log


def log_norm(dataframe, par_log):
    # num_total = len(dataframe)
    for i in par_log:
        temp = dataframe[i].tolist()
        min_data_abs = abs(min(temp))
        temp_log = [log(min_data_abs + 1 + j) for j in temp]
        temp_name = i + "_log"
        dataframe[temp_name] = temp_log
    return dataframe


def Min_Max(normalization_parameter, file_dataframe):
    """
    This function is used to data standardization.

    Parameters
    ----------
    normalization_parameter : list
    The attribute we want to normalize.
    file_dataframe : DataFrame
    The DataFrame consists of key attributes.

    Returen
    -------
    file_dataframe : DataFrame
    The DataFrame finished normalized.
    """
    normalization_parameter_result = []
    for i in normalization_parameter:
        par = file_dataframe[i].tolist()
        Max_par = max(par)
        Min_par = min(par)
        div = Max_par - Min_par
        if div == 0:
            if Max_par == 0:
                norm = par
            else:
                norm = [j / Max_par for j in par]
        else:
            norm = [round((j - Min_par) / div, 6) for j in par]
        normalization_parameter_result.append(norm)
    for j in range(len(normalization_parameter)):
        norm_name = normalization_parameter[j] + '_norm'
        file_dataframe[norm_name] = normalization_parameter_result[j]
    # file_dataframe_norm = file_dataframe.drop(normalization_parameter, axis=1)
    return file_dataframe


def min_fee(dataframe, para_fee):
    fee_array = np.array(dataframe[para_fee])
    fee_min = np.min(fee_array, axis=1)
    dataframe["fee_min"] = fee_min
    return dataframe


def traffic_current_month(dataframe):
    month_traffic = dataframe["month_traffic"].tolist()
    last_month_traffic = dataframe["last_month_traffic"].tolist()
    num = len(month_traffic)
    traffic_current_month = [month_traffic[i] - last_month_traffic[i] for i in range(num)]
    dataframe["traffic_current_month"] = traffic_current_month
    return dataframe


train_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\train4_feature.csv",
                       encoding="utf-8", low_memory=False)
test_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\test4_feature.csv",
    encoding="utf-8", low_memory=False)

# train_data = train_data.iloc[0:100]
# test_data = test_data.iloc[0:100]

train_service_type = train_data["current_service"].tolist()
train_service_type_encode = train_data["service_type_encode"].tolist()
train_data_encode = train_data.drop(['current_service', 'service_type_encode'], axis=1)
num_train = len(train_data)
combine_data_1 = pd.concat([train_data_encode, test_data])
min_fee_df = min_fee(combine_data_1, ['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee'])
combine_data = traffic_current_month(min_fee_df)
num_total = len(combine_data)
column_name = train_data.columns.values.tolist()
column_name_norm = [i for i in column_name if i.find('_norm') != -1]
column_name_log = [i[:-5] for i in column_name_norm]
column_name_log = column_name_log + ['fee_min', 'traffic_current_month']
log_df = log_norm(combine_data, column_name_log)
final_df = Min_Max(['fee_min', 'traffic_current_month'], log_df)
train_log = final_df.iloc[0:num_train]
test_log = final_df.iloc[num_train:num_total]
train_log["current_service"] = train_service_type
train_log["service_type_encode"] = train_service_type_encode
train_log.to_csv(r'E:\CCFDF\plansmatching\data\raw data\final_data\train_feature_2.csv', index=False)
test_log.to_csv(r'E:\CCFDF\plansmatching\data\raw data\final_data\test_feature_2.csv', index=False)

