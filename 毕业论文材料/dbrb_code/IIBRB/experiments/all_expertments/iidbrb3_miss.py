
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from IIBRB.experiments.random_process import random_array
from IIBRB.iidbrb3 import IIDBRBClassifier
from datasets.load_data import *
from datasets.output_excel import write_data_to_excel
from datasets.process_data import process_to_pieces


def cross_validation():
    N_SPLITS = 10
    delta = 1/16
    X, y = load_banknote()

    A, D = process_to_pieces(X, y, 2)
    kf = KFold(n_splits=N_SPLITS, shuffle=True)
    dbrb = IIDBRBClassifier(A, D, delta)
    maes = []
    for train_index, test_index in kf.split(X):
        train_X, train_y = X[train_index, :], y[train_index]
        test_X, test_y = X[test_index], y[test_index]
        dbrb.fit(train_X, train_y)
        print('++load_banknote ++single_miss ++%s ++delta=1/%d' % (N_SPLITS, (delta ** -1)))
        row = []
        # 在单个属性集上缺失数据
        for i in range(np.shape(A)[0]):
            tmp_X = np.copy(test_X)
            tmp_X[:, i] = np.full(len(tmp_X), np.nan)
            y_predict = dbrb.predict(tmp_X)
            row.append(accuracy_score(y_predict, test_y))
        maes.append(row)
        print("acc: %s\n" % row)
    return maes


total_acc = []
avg_acc = 0
for tc in range(10):
    print("-------------------我是分割线(%d)--------------------------" % (tc + 1))
    maes = cross_validation()
    acc, std = np.mean(maes, 0), np.std(maes, 0)
    total_acc.append(acc)
    print("mean acc is :")
    print(np.mean(total_acc, 0))
    print("best acc is:")
    print(np.max(total_acc, 0))
    print("")
    # write_data_to_excel("iidbrb3_iris_"+str(tc+1), maes)
