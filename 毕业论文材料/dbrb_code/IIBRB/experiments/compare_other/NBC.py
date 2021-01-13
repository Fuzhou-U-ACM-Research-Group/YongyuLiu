
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB

from IIBRB.far import FAR
from datasets.load_data import *
from datasets.output_excel import write_data_to_excel
from datasets.process_data import process_to_pieces
import pandas as pd
import impyute as impy


def MICE(complete_data, incomplete_data):
    start_idx = len(complete_data)
    data = np.concatenate((complete_data, incomplete_data))
    data_mice = impy.mice(data)
    return data_mice[start_idx:]


def cross_validation():
    # X, y = load_breast_cancer()
    # X = X[:, 1:]
    # incomplete_X = []
    # incomplete_y = []
    # complete_X = []
    # complete_y = []
    # for i in range(np.shape(X)[0]):
    #     if pd.isnull(X[i]).sum() != 0:
    #         incomplete_X.append(X[i])
    #         incomplete_y.append(y[i])
    #     else:
    #         complete_X.append(X[i])
    #         complete_y.append(y[i])
    # X, y = np.array(complete_X), np.array(complete_y)
    X, y = load_banknote()
    # X, y = load_transfusion()
    clf = GaussianNB()
    maes = []
    N_SPLITS = 10
    kf = KFold(n_splits=N_SPLITS, shuffle=True)
    for train_index, test_index in kf.split(X):
        train_X, train_y = X[train_index, :], y[train_index]
        test_X, test_y = X[test_index], y[test_index]
        clf.fit(train_X, train_y)
        far = FAR(train_X)
        row = []
        # 在单个属性集上缺失数据
        for i in range(np.shape(X)[1]):
            tmp_X = np.copy(test_X)
            fill_val = np.mean(train_X[:, i])
            tmp_X[:, i] = np.full(len(tmp_X), np.nan)
            tmp_X = far.fill_in(tmp_X)
            # tmp_X = MICE(train_X, tmp_X)
            y_predict = clf.predict(tmp_X)
            row.append(accuracy_score(y_predict, test_y))
        maes.append(row)
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
    print("total mean acc: %s" % np.mean(np.mean(total_acc, 0)))
    print('++ iris 10CV far_fill_in NBC')
# ==== iris ====
# 0.730666665
# ==== breast cancer ====
# 0.9629930377948281

