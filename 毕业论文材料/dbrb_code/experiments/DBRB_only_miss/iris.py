import random

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from DBRB_opt.dbrb import DBRBClassifier
from datasets.load_data import load_wine
from datasets.process_data import process_to_pieces


def cross_validation():
    N_SPLITS = 10
    X, y = load_wine()
    A, D = process_to_pieces(X, y, 3)
    kf = KFold(n_splits=N_SPLITS, shuffle=True)
    dbrb = DBRBClassifier(A, D)
    maes = []
    times = []
    for train_index, test_index in kf.split(X):
        train_X, train_y = X[train_index, :], y[train_index]
        test_X, test_y = X[test_index], y[test_index]
        dbrb = dbrb.fit(train_X, train_y)
        y_predict = dbrb.predict(test_X)
        maes.append(accuracy_score(y_predict, test_y))
        times.append(dbrb.average_process_time)
    return np.mean(maes), np.std(maes), np.mean(times)


'''
缺失单个属性的数据
10次10CV，取平均和最高准确率
每次训练完模型后，测试缺失每个属性后的准确率，然后取均值
'''
best_acc, best_std, best_time = 0, 0, 0
avg_acc = 0
for tc in range(10):
    print("-------------------我是分割线(%d)--------------------------" % (tc + 1))
    acc, std, time = cross_validation()
    avg_acc = (tc * avg_acc + acc) / (tc + 1)
    if acc > best_acc:
        best_acc, best_std = acc, std
    print("acc:%f(std:%f)" % (acc, std))
    print("best_acc:%f(std:%f)" % (best_acc, best_std))
    print("avg_acc:%f" % avg_acc)
    print("")
