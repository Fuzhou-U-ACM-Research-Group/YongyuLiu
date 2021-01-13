
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from IIBRB.experiments.random_process import random_array
from IIBRB.iidbrb3 import IIDBRBClassifier
from datasets.load_data import load_iris,load_transfusion
from datasets.output_excel import write_data_to_excel
from datasets.process_data import process_to_pieces


def cross_validation():
    N_SPLITS = 10
    X, y = load_transfusion()

    A, D = process_to_pieces(X, y, 2)
    kf = KFold(n_splits=N_SPLITS, shuffle=True)
    dbrb = IIDBRBClassifier(A, D)
    maes = []
    for train_index, test_index in kf.split(X):
        train_X, train_y = X[train_index, :], y[train_index]
        test_X, test_y = X[test_index], y[test_index]
        dbrb.fit(train_X, train_y)
        row = []
        # 全局随机缺失
        # tmp_X = np.copy(test_X)
        # percent = [0.1, 0.3, 0.5, 0.7, 0.9]
        # for p in percent:
        #     miss_X = random_array(tmp_X, p)
        #     y_predict = dbrb.predict(miss_X)
        #     row.append(accuracy_score(y_predict, test_y))
        # 在单个属性集上缺失数据
        for i in range(np.shape(A)[0]):
            tmp_X = np.copy(test_X)
            tmp_X[:, i] = np.full(len(tmp_X), np.nan)
            y_predict = dbrb.predict(tmp_X)
            row.append(accuracy_score(y_predict, test_y))
        # miss_idx = [[0, 2], [0, 1], [1, 3]]
        # for idx in range(len(miss_idx)):
        #     tmp_X = np.copy(test_X)
        #     tmp_X[:, miss_idx[idx][0]] = np.full(len(tmp_X), np.nan)
        #     tmp_X[:, miss_idx[idx][1]] = np.full(len(tmp_X), np.nan)
        #     y_predict = dbrb.predict(tmp_X)
        #     row.append(accuracy_score(y_predict, test_y))
        maes.append(row)
    return maes


total_acc = []
avg_acc = 0
for tc in range(10):
    print("-------------------我是分割线(%d)--------------------------" % (tc + 1))
    maes = cross_validation()
    acc, std = np.mean(maes, 0), np.std(maes, 0)
    total_acc.append(acc)
    print("++ 10CV multi")
    print("mean acc is :")
    print(np.mean(total_acc, 0))
    print("best acc is:")
    print(np.max(total_acc, 0))
    print("")
    # write_data_to_excel("iidbrb3_iris_"+str(tc+1), maes)
