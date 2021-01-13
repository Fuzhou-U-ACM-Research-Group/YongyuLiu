
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from IIBRB.experiments.random_process import random_array
from IIBRB.iidbrb3 import IIDBRBClassifier
from IIBRB.iidbrb2 import IIDBRBClassifier2
from datasets.load_data import load_iris
from datasets.process_data import process_to_pieces


def cross_validation():
    N_SPLITS = 10
    X, y = load_iris()

    A, D = process_to_pieces(X, y, 3)
    kf = KFold(n_splits=N_SPLITS, shuffle=True)
    dbrb = IIDBRBClassifier(A, D, delta=1/30)
    maes = []
    for train_index, test_index in kf.split(X):
        train_X, train_y = X[train_index, :], y[train_index]
        test_X, test_y = X[test_index], y[test_index]
        dbrb = dbrb.fit(train_X, train_y)
        y_predict = dbrb.predict(test_X)
        maes.append(accuracy_score(y_predict, test_y))
    return np.mean(maes), np.std(maes)


total_acc = []
avg_acc = 0
for tc in range(10):
    print("-------------------我是分割线(%d)--------------------------" % (tc + 1))
    acc, std = cross_validation()
    total_acc.append(acc)
    print("mean acc is :")
    print(np.mean(total_acc, 0))
    print("best acc is:")
    print(np.max(total_acc, 0))
    print("Gauss fuzz function ++delta: 1/30 ")
# mean acc is :
# 0.9433333333333331
# best acc is:
# 0.9666666666666666
# gause fuzzy function 1/8

# mean acc is :
# 0.9446666666666668
# best acc is:
# 0.9666666666666666
# Gauss fuzz function ++delta: 1/16

# mean acc is :
# 0.9313333333333332
# best acc is:
# 0.9466666666666667
# Gauss fuzz function ++delta: 1/3

# mean acc is :
# 0.9460000000000001
# best acc is:
# 0.9533333333333334
# Gauss fuzz function ++delta: 1/2

# mean acc is :
# 0.9346666666666668
# best acc is:
# 0.9466666666666667
# Gauss fuzz function ++delta: 1/4

# mean acc is :
# 0.9486666666666667
# best acc is:
# 0.96
# change nothing

# mean acc is :
# 0.9560000000000002
# best acc is:
# 0.9666666666666666
# Gauss fuzz function ++delta: 1/20

# mean acc is :
# 0.9526666666666669
# best acc is:
# 0.9600000000000002
# Gauss fuzz function ++delta: 1/30

# mean acc is :
# 0.9526666666666668
# best acc is:
# 0.9600000000000002
# Gauss fuzz function ++delta: 1/50
