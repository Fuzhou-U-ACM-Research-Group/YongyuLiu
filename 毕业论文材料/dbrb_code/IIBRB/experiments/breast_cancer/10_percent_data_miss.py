from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from IIBRB.iidbrb3 import IIDBRBClassifier
# from IIBRB.iidbrb import IIDBRBClassifier
from datasets.load_data import load_breast_cancer
import numpy as np
import pandas as pd

from datasets.output_excel import write_data_to_excel
from datasets.process_data import process_to_pieces

X, y = load_breast_cancer()
X = X[:, 1:]
incomplete_X = []
incomplete_y = []
complete_X = []
complete_y = []
for i in range(np.shape(X)[0]):
    if pd.isnull(X[i]).sum() != 0:
        incomplete_X.append(X[i])
        incomplete_y.append(y[i])
    else:
        complete_X.append(X[i])
        complete_y.append(y[i])

complete_X = np.array(complete_X)
complete_y = np.array(complete_y)
kf = KFold(n_splits=10, shuffle=True)
maes = []
for train_index, test_index in kf.split(complete_X):
    train_X, train_y = complete_X[train_index, :], complete_y[train_index]
    test_X, test_y = complete_X[test_index], complete_y[test_index]
    A, D = process_to_pieces(train_X, train_y, 2)
    dbrb = IIDBRBClassifier(A, D)
    dbrb = dbrb.fit(train_X, train_y)
    # 缺失第i个属性的数据
    row = []
    # for i in range(np.shape(A)[0]):
    #     tmp_X = np.copy(test_X)
    #     tmp_X[:, i] = np.full(len(tmp_X), np.nan)
    #     y_predict = dbrb.predict(tmp_X)
    #     row.append(accuracy_score(y_predict, test_y))
    # 缺失多个属性值
    miss_idx = [[0, 3, 8], [1, 4, 6], [2, 5, 7]]
    for idx in range(len(miss_idx)):
        tmp_X = np.copy(test_X)
        tmp_X[:, miss_idx[idx][0]] = np.full(len(tmp_X), np.nan)
        tmp_X[:, miss_idx[idx][1]] = np.full(len(tmp_X), np.nan)
        tmp_X[:, miss_idx[idx][2]] = np.full(len(tmp_X), np.nan)
        y_predict = dbrb.predict(tmp_X)
        row.append(accuracy_score(y_predict, test_y))
    maes.append(row)

print(maes)
print('行均值：')
print(np.mean(maes, 1))
print('列均值：')
print(np.mean(maes, 0))
write_data_to_excel("iidbrb3_muti_miss_breast_canner", maes)
