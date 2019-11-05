import pandas as pd
import numpy as np


def similarity_caculate_weight(train_data, test_data, para_list, weight):
    calculate_test_data = test_data[para_list]
    calculate_train_data = train_data[para_list]
    weight_new = [tt ** 2 for tt in weight]
    train_num = len(calculate_train_data)
    test_num = len(calculate_test_data)
    weight_array = np.array([weight_new] * train_num)
    train_array = np.array(calculate_train_data)
    train_weight_array = train_array * weight_array
    train_userid = train_data["user_id"].tolist()
    train_type = train_data["current_service"].tolist()
    # distance = []
    # index = []
    distance_top10 = []
    index_top10 = []
    # user_id = []
    user_id_top10 = []
    type_top10 = []
    for i in range(test_num):
        temp = calculate_test_data.iloc[i].tolist()
        temp_weight_array = np.array(temp * weight_new)
        temp_distance = [round(np.linalg.norm(temp_weight_array - train_weight_array[j]), 6) for j in range(train_num)]
        index_temp_distange = list(range(len(temp_distance)))
        index_temp_distange.sort(key=lambda m: temp_distance[m])
        temp_distance.sort()
        # distance.append(temp_distance)
        # index.append(index_temp_distange)
        distance_top10.append(temp_distance[0:10])
        index_top10.append(index_temp_distange[0:10])
        user_id_top10.append([train_userid[n] for n in index_temp_distange[0:10]])
        type_top10.append([train_type[n] for n in index_temp_distange[0:10]])
    distance_top10 = pd.DataFrame(np.array(distance_top10))
    index_top10 = pd.DataFrame(np.array(index_top10))
    user_id_top10 = pd.DataFrame(np.array(user_id_top10))
    type_top10 = pd.DataFrame(np.array(type_top10))
    return distance_top10, index_top10, user_id_top10, type_top10


def similarity_caculate(train_data, test_data, para_list):
    calculate_test_data = test_data[para_list]
    calculate_train_data = train_data[para_list]
    train_num = len(calculate_train_data)
    test_num = len(calculate_test_data)
    train_array = np.array(calculate_train_data)
    train_userid = train_data["user_id"].tolist()
    train_type = train_data["current_service"].tolist()
    # distance = []
    # index = []
    distance_top10 = []
    index_top10 = []
    # user_id = []
    user_id_top10 = []
    type_top10 = []
    for i in range(test_num):
        temp = calculate_test_data.iloc[i].tolist()
        temp_weight_array = np.array(temp)
        temp_distance = [round(np.linalg.norm(temp_weight_array - train_array[j]), 6) for j in range(train_num)]
        index_temp_distange = list(range(len(temp_distance)))
        index_temp_distange.sort(key=lambda m: temp_distance[m])
        temp_distance.sort()
        # distance.append(temp_distance)
        # index.append(index_temp_distange)
        distance_top10.append(temp_distance[0:10])
        index_top10.append(index_temp_distange[0:10])
        user_id_top10.append([train_userid[n] for n in index_temp_distange[0:10]])
        type_top10.append([train_type[n] for n in index_temp_distange[0:10]])
    distance_top10 = pd.DataFrame(np.array(distance_top10))
    index_top10 = pd.DataFrame(np.array(index_top10))
    user_id_top10 = pd.DataFrame(np.array(user_id_top10))
    type_top10 = pd.DataFrame(np.array(type_top10))
    return distance_top10, index_top10, user_id_top10, type_top10


data_train = pd.read_csv(r"/data/projects/CCFDF_18/data/train_combine_4_encode_precentage.csv", encoding="utf-8",
                         low_memory=False)
data_test = pd.read_csv(r"/data/projects/CCFDF_18/data/test_4_combine.csv", encoding="utf-8", low_memory=False)

# data_train = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\train_combine_4_encode_precentage.csv", encoding="utf-8",
#                          low_memory=False)
# data_test = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\test_4_combine.csv", encoding="utf-8", low_memory=False)
# data_test = data_test.iloc[0:20]

parameter_list = ['service1_caller_time_precentage', 'service1_caller_time_norm',
                  'fee_std_norm', 'online_time_norm', 'contract_time_precentage',
                  'month_traffic_norm', 'contract_time_norm',
                  'service2_caller_time_precentage', 'service_caller_time_fluctuate_norm',
                  'contract_type', 'service2_caller_time_norm', '4_total_fee_norm',
                  ]

weight_list = [0.189179, 0.164586, 0.130344, 0.083995, 0.073212, 0.070564, 0.065456, 0.051835, 0.046727, 0.044457,
               0.041998, 0.037647]
weight_list = weight_list[::-1]
distance_top10_df, index_top10_df, user_id_top10_df, type_top10_df = similarity_caculate(data_train,
                                                                                         data_test,
                                                                                         parameter_list,
                                                                                         )
distance_top10_df.to_csv(r"/data/projects/CCFDF_18/result/RF/distance_top10.csv")
index_top10_df.to_csv(r"/data/projects/CCFDF_18/result/RF/index_top10.csv")
user_id_top10_df.to_csv(r"/data/projects/CCFDF_18/result/RF/user_id_top10.csv")
type_top10_df.to_csv(r"/data/projects/CCFDF_18/result/RF/type_top10.csv")


# distance_top10_df.to_csv(r"C:\Users\poder\Desktop\distance_top10.csv")
# index_top10_df.to_csv(r"C:\Users\poder\Desktop\index_top10.csv")
# user_id_top10_df.to_csv(r"C:\Users\poder\Desktop\user_id_top10.csv")
# type_top10_df.to_csv(r"C:\Users\poder\Desktop\type_top10.csv")