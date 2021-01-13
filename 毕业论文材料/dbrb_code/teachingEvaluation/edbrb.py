from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge

from IIBRB.dbrb import DBRBRegressor
from EDBRB.ebrb import EDBRBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import math
import numpy as np

from datasets.load_data import load_csv_by
from teachingEvaluation.set_pipline import process_to_pieces
from teachingEvaluation.set_pipline import data_cleansing
from teachingEvaluation.model import *


def cross_validation():
    data = load_csv_by("jscp_201801_notAll_data_set.csv", path='data/student_evaluation_teacher')
    data2 = load_csv_by("jscp_201801_all_data_set.csv", path='data/student_evaluation_teacher')
    data = np.concatenate((data, data2), axis=0)
    X, y = data[:, :-1], data[:, -1]
    X, y = data_cleansing(X, y)
    A, D = process_to_pieces(X, y)
    ebrb = EDBRBRegressor(A, D)
    means = [[]]
    N_SPLITS = 2
    kf = KFold(n_splits=N_SPLITS, shuffle=True)
    for train_index, test_index in kf.split(X):
        train_X, train_y = X[train_index, :], y[train_index]
        test_X, test_y = X[test_index], y[test_index]

        e = ebrb_model(ebrb, train_X, train_y, test_X, test_y)
        means[0].append(e)
    means = np.array(means)

    eval = ['RMSE', 'MAE', 'R2', 'R2_adjust', 'EVS']
    method_name = ['ebrb']
    for i in range(len(eval)):
        print("======== " + eval[i] + " ========")
        for j in range(len(means)):
            print(method_name[j] + ":" + str(np.mean(means[j, :, i])))

    index = np.arange(1, np.shape(means)[1]+1, 1)

    # for i in range(len(eval)):
    #     plt.figure()
    #     plt.plot(index, means[0, :, i], color='black', linestyle='-', label='ebrb')
    #     # plt.plot(index, means[1, :, i], color='black', linestyle='-', label='brb')
    #     plt.ylim()
    #     # plt.title("mean absolute error")
    #     plt.xlabel("count")
    #     plt.ylabel(eval[i])
    #     plt.legend()
    #     plt.savefig(eval[i] + '.png')
    #
    # plt.show()
    return


cross_validation()
