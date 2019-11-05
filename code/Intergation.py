import pandas as pd
import numpy as np
import matplotlib.pyplot as plot



def revelant_map(select_dataframe):
    cormat = pd.DataFrame(select_dataframe.corr())
    plot.pcolor(cormat)
    plot.show()
    return cormat



result_1 = pd.read_csv(r"/Users/peterlee/Documents/CCFDF18/final_data/result_test/XGBoost_learning_rate.csv",
                       encoding="utf-8", low_memory=False)["current_service"].tolist()
result_2 = pd.read_csv(r"/Users/peterlee/Documents/CCFDF18/final_data/result_test/XGBoost_balance.csv",
                       encoding="utf-8", low_memory=False)["current_service"].tolist()
result_3 = pd.read_csv(r"/Users/peterlee/Documents/CCFDF18/final_data/result_test/XGBoost_unoptimization.csv",
                       encoding="utf-8", low_memory=False)["current_service"].tolist()
result_6 = pd.read_csv(r"/Users/peterlee/Documents/CCFDF18/final_data/result_test/result_precentage.csv",
                       encoding="utf-8", low_memory=False)["current_service"].tolist()
result_7 = pd.read_csv(r"/Users/peterlee/Documents/CCFDF18/final_data/result_test/result_testnum_0.03.csv",
                       encoding="utf-8", low_memory=False)["current_service"].tolist()
result_11 = pd.read_csv(r"/Users/peterlee/Documents/CCFDF18/final_data/result_test/result_combine_2.csv",
                        encoding="utf-8", low_memory=False)["current_service"].tolist()
result_13 = pd.read_csv(r"/Users/peterlee/Documents/CCFDF18/final_data/result_test/my_sample_6.csv",
                        encoding="utf-8", low_memory=False)["current_service"].tolist()
result_15 = pd.read_csv(r"/Users/peterlee/Documents/CCFDF18/final_data/result_test/my_sample_4.csv",
                        encoding="utf-8", low_memory=False)["current_service"].tolist()
result_16 = pd.read_csv(r"/Users/peterlee/Documents/CCFDF18/final_data/result_test/my_sample_3.csv",
                        encoding="utf-8", low_memory=False)["current_service"].tolist()

F1_list = [0.73093957000, 0.71842682000, 0.73351532000,
           0.70677644000, 0.70832336000, 0.69593376000,
           0.67634636000, 0.66584486000, 0.64061570000]
F1_list = [2.8, 2.5, 2.8, 2, 2, 2, 1.8, 1.5, 1.3]
F_1_sum = sum(F1_list)
weight_vote = [round(i/F_1_sum, 6) for i in F1_list]
result_combine = np.array([result_1, result_2, result_3, result_6, result_7, result_11, result_13,
                           result_15, result_16]).transpose()
revelant = np.array(revelant_map(pd.DataFrame(result_combine)))
sum_re = np.sum(revelant, axis=1)

service_type = [89950166, 89950167, 89950168, 99999825, 99999826, 99999827, 99999828, 99999830, 90063345, 90109916, 90155946]
num_total = len(result_1)
num_class = len(weight_vote)
num_service_type = len(service_type)
vote = []
for i in range(num_total):
    # vote_row = []
    # temp = result_combine[i]
    # for j in service_type:
    #     type_temp_list = [k for k in range(num_class) if j == temp[k]]
    #     if len(type_temp_list) == 0:
    #         vote_row.append(0)
    #     else:
    #         vote_row.append(sum([weight_vote[m] for m in type_temp_list]))
    vote_row = [0] * num_service_type
    temp = result_combine[i]
    for j in range(num_class):
        index_type = service_type.index(temp[j])
        vote_row[index_type] += weight_vote[j]
    vote.append(vote_row)
vote_result = []
for i in range(num_total):
    temp_index = vote[i].index(max(vote[i]))
    vote_result.append(service_type[temp_index])

user_id =  pd.read_csv(r"/Users/peterlee/Documents/CCFDF18/final_data/result_test/my_sample_3.csv",
                        encoding="utf-8", low_memory=False)["user_id"].tolist()
final_da = pd.DataFrame({"user_id": user_id, "current_service": vote_result})
final_da.to_csv(r"/Users/peterlee/Documents/CCFDF18/final_data/result_test/vote_weightscore.csv")




