import pandas as pd


# train = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\train_all\train_all _duplicate.csv", encoding="utf-8",
#                     low_memory=False)
# test = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\republish_test\republish_test_1.csv", encoding="utf-8",
#                     low_memory=False)
# combine = pd.concat([train, test])
# combine.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\combine.csv")
# raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\combine.csv", encoding="utf-8", low_memory=False)
# type_service = raw_data["service_type"].tolist()
# num = len(type_service)
# service_type_4 = [i for i in range(num) if type_service[i] == 4]
# dataframe_4 = raw_data.iloc[service_type_4]
# dataframe_4.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\combine_4.csv")

raw_data = pd.read_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\train_combine_4_new.csv", encoding="utf-8",
                       low_memory=False)

service_type = raw_data["current_service"].tolist()
new_type = []
for i in service_type:
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
    # if i == 90063345:
    #     new_type.append(8)
    # if i == 90109916:
    #     new_type.append(9)
    # if i == 90155946:
    #     new_type.append(10)
# raw_data["service_type_encode"] = new_type
# raw_data.to_csv(r"E:\CCFDF\plansmatching\data\raw data\final_data\train_combine_4_encode.csv")


