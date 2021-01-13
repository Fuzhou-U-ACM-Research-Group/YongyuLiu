
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from IIBRB.experiments.random_process import random_array
from IIBRB.iidbrb3 import IIDBRBClassifier
from IIBRB.iidbrb2 import IIDBRBClassifier2
from datasets.load_data import *
from datasets.process_data import process_to_pieces


def cross_validation(detal=1/8):
    N_SPLITS = 10
    X, y = load_transfusion()
    A, D = process_to_pieces(X, y, 2)
    kf = KFold(n_splits=N_SPLITS, shuffle=True)
    dbrb = IIDBRBClassifier(A, D, delta=detal)
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
    delta = 49/50
    acc, std = cross_validation(delta)
    total_acc.append(acc)
    print("mean acc is :")
    print(np.mean(total_acc, 0))
    print("best acc is:")
    print(np.max(total_acc, 0))
    print("delta = %s" % delta)


