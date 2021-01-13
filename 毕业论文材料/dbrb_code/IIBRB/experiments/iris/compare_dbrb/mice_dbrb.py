
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from IIBRB.dbrb import DBRBClassifier
from datasets.load_data import load_iris
from datasets.output_excel import write_data_to_excel
from datasets.process_data import process_to_pieces
import impyute as impy


def MICE(complete_data, incomplete_data):
    start_idx = len(complete_data)
    data = np.concatenate((complete_data, incomplete_data))
    data_mice = impy.mice(data)
    return data_mice[start_idx:]

def cross_validation():
    N_SPLITS = 2
    X, y = load_iris()

    A, D = process_to_pieces(X, y, 3)
    kf = KFold(n_splits=N_SPLITS, shuffle=True)
    dbrb = DBRBClassifier(A, D)
    maes = []
    for train_index, test_index in kf.split(X):
        train_X, train_y = X[train_index, :], y[train_index]
        test_X, test_y = X[test_index], y[test_index]
        dbrb.fit(train_X, train_y)
        row = []
        # 在单个属性集上缺失数据
        for i in range(np.shape(A)[0]):
            tmp_X = np.copy(test_X)
            tmp_X[:, i] = np.full(len(tmp_X), np.nan)
            tmp_X = MICE(train_X, tmp_X)
            y_predict = dbrb.predict(tmp_X)
            row.append(accuracy_score(y_predict, test_y))
        # miss_idx = [[0, 2], [0, 1], [1, 3]]
        # for idx in range(len(miss_idx)):
        #     tmp_X = np.copy(test_X)
        #     tmp_X[:, miss_idx[idx][0]] = np.full(len(tmp_X), np.nan)
        #     tmp_X[:, miss_idx[idx][1]] = np.full(len(tmp_X), np.nan)
        #     tmp_X = MICE(train_X, tmp_X)
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
    print("mean acc is :")
    print(np.mean(total_acc, 0))
    print("best acc is:")
    print(np.max(total_acc, 0))
    print("")
    write_data_to_excel("2CV_mice_dbrb_iris_"+str(tc+1), maes)
# ==== single =====
# 10CV
# mean acc is :
# [0.94533333 0.94666667 0.93733333 0.93866667]
# best acc is:
# [0.95333333 0.95333333 0.94666667 0.96      ]
#  5CV
# mean acc is :
# [0.948      0.95       0.94266667 0.93933333]
# best acc is:
# [0.96666667 0.97333333 0.95333333 0.96666667]
#  2CV
# mean acc is :
# [0.946      0.94933333 0.94133333 0.93533333]
# best acc is:
# [0.95333333 0.96666667 0.96       0.95333333]
# ==== multi ====
# 2CV
# mean acc is :
# [0.89466667 0.95066667 0.93466667]
# best acc is:
# [0.96       0.96666667 0.96      ]
# 5CV
# mean acc is :
# [0.92466667 0.954      0.93933333]
# best acc is:
# [0.96       0.96666667 0.95333333]


