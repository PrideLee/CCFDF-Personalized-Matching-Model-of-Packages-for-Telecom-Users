# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 21:57:25 2018

@author: vvv
"""
import csv
import numpy as np
from sklearn.metrics import confusion_matrix

SERVICE = ['89950166', '89950167', '89950168', '90063345',
           '90109916', '90155946', '99999825', '99999826',
           '99999827', '99999828', '99999830']

test_file = 'valid.csv'

predict_file = 'my_test.csv'


def read_data(filename):
    data = []
    with open(filename) as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)
        for row in csv_reader:
            if row[1] not in SERVICE:
                raise ValueError("Check you label name. "
                                 "%s is an error uid:" % row[1])
            data.append(row[1])
    return data


y_pred = read_data(predict_file)  # read predict
y_true = read_data(test_file)  # read test
print("-----------Finish reading-------------")

confusion_max = confusion_matrix(y_true=y_true,
                                 y_pred=y_pred,
                                 labels=SERVICE)
# Return [1,NUM_SERVICE] array
TP = confusion_max.diagonal()
FP = confusion_max.sum(0) - TP
FN = confusion_max.sum(1) - TP
multi_f1 = 2 * TP / (FP + FN + 2 * TP)
print('The F1 score of each service:', multi_f1)
F1 = np.square(multi_f1.mean())
print('The total F1 score is %4f' % F1)
