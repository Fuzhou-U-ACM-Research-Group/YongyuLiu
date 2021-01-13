
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from IIBRB.experiments.random_process import random_array
from IIBRB.iidbrb3 import IIDBRBClassifier
from datasets.load_data import load_wine
from datasets.output_excel import write_data_to_excel
from datasets.process_data import process_to_pieces


def cross_validation():
    N_SPLITS = 10
    X, y = load_wine()

    A, D = process_to_pieces(X, y, 3)
    kf = KFold(n_splits=N_SPLITS, shuffle=True)
    dbrb = IIDBRBClassifier(A, D)
    maes = []
    for train_index, test_index in kf.split(X):
        train_X, train_y = X[train_index, :], y[train_index]
        test_X, test_y = X[test_index], y[test_index]
        dbrb.fit(train_X, train_y)
        y_predict = dbrb.predict(test_X)
        maes.append(accuracy_score(y_predict, test_y))
    return maes


total_acc = []
avg_acc = 0
for tc in range(10):
    print("-------------------我是分割线(%d)--------------------------" % (tc + 1))
    maes = cross_validation()
    acc = np.mean(maes)
    total_acc.append(acc)
    print("mean acc is :")
    print(np.mean(total_acc))
    print("best acc is:")
    print(np.max(total_acc))
    print("")
    # write_data_to_excel("iidbrb3_iris_"+str(tc+1), maes)
