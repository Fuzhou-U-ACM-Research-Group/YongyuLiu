
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from IIBRB.dbrb import DBRBClassifier
from datasets.load_data import load_iris
from datasets.output_excel import write_data_to_excel
from datasets.process_data import process_to_pieces


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
            fill_val = np.mean(train_X[:, i])
            tmp_X[:, i] = np.full(len(tmp_X), fill_val)
            y_predict = dbrb.predict(tmp_X)
            row.append(accuracy_score(y_predict, test_y))
        # miss_idx = [[0, 2], [0, 1], [1, 3]]
        # for idx in range(len(miss_idx)):
        #     tmp_X = np.copy(test_X)
        #     fill_val_0 = np.mean(train_X[:, miss_idx[idx][0]])
        #     fill_val_1 = np.mean(train_X[:, miss_idx[idx][1]])
        #     tmp_X[:, miss_idx[idx][0]] = np.full(len(tmp_X), fill_val_0)
        #     tmp_X[:, miss_idx[idx][1]] = np.full(len(tmp_X), fill_val_1)
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
    write_data_to_excel("2CV_mean_dbrb_iris_"+str(tc+1), maes)
# 10CV
# mean acc is :
# [0.956      0.95133333 0.53333333 0.618     ]
# best acc is:
# [0.97333333 0.96666667 0.62666667 0.70666667]
#  5CV
# mean acc is :
# [0.95266667 0.95       0.58666667 0.584     ]
# best acc is:
# [0.97333333 0.96666667 0.7        0.70666667]
#  2CV
# mean acc is :
# [0.94266667 0.94466667 0.658      0.592     ]
# best acc is:
# [0.96666667 0.96       0.78666667 0.86      ]
# ==== multi ====
# # 2CV
# # mean acc is :
# # [0.56  0.948 0.588]
# # best acc is:
# # [0.76666667 0.97333333 0.69333333]
# # 5CV
# # mean acc is :
# # [0.50266667 0.95266667 0.56133333]
# # best acc is:
# # [0.62666667 0.98       0.76      ]
