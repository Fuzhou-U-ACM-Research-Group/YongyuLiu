from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge

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
    # X, y = data_cleansing_nan(X, y)
    X, y = data_cleansing(X, y)
    A, D = process_to_pieces(X, y)
    edbrb = EDBRBRegressor(A, D)
    dtc = DecisionTreeRegressor()
    svr = SVR()
    mlp = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2))
    bys = BayesianRidge()
    means = [[], [], [], [], []]
    N_SPLITS = 10
    kf = KFold(n_splits=N_SPLITS, shuffle=True)
    for train_index, test_index in kf.split(X):
        train_X, train_y = X[train_index, :], y[train_index]
        test_X, test_y = X[test_index], y[test_index]
        a = svr_regression_model(svr, train_X, train_y, test_X, test_y)
        b = decision_tree_model(dtc, train_X, train_y, test_X, test_y, [6])
        c = mlp_model(mlp, train_X, train_y, test_X, test_y)
        d = bys_model(bys, train_X, train_y, test_X, test_y)
        e = ebrb_model(edbrb, train_X, train_y, test_X, test_y)
        means[0].append(a)
        means[1].append(b)
        means[2].append(c)
        means[3].append(d)
        means[4].append(e)
    means = np.array(means)

    eval = ['MAE', 'RMSE', 'R2', 'R2_adjust', 'EVS']
    method_name = ['svr', 'dtc', 'mlp', 'bys', 'e-dbrb']
    max_value = []
    for i in range(len(eval)):
        print("======== " + eval[i] + " ========")
        mv = 0
        for j in range(len(means)):
            if mv < np.max(means[j, :, i]):
                mv = np.max(means[j, :, i])
            print(method_name[j] + ":" + str(np.mean(means[j, :, i])))
        max_value.append(mv)

    index = np.arange(1, np.shape(means)[1]+1, 1)

    for i in range(len(eval)):
        plt.figure()
        plt.bar(index, means[1, :, i], width=1/5, label='svr')
        if i != 2 and i != 3 and i != 4:
            plt.bar(index+1/5, means[2, :, i], width=1/5, label='dtc')
            plt.bar(index+2/5, means[3, :, i], width=1/5, label='mlp')
        plt.bar(index+3/5, means[0, :, i], width=1/5, label='e-dbrb')
        # plt.bar(index, means[4, :, i], color='red', linestyle='-', label='bys')
        plt.ylim()
        # plt.title("mean absolute error")
        plt.xlabel("10 fold cross validation")
        plt.ylabel(eval[i])
        plt.ylim((0, max_value[i] + 1))
        plt.legend()
        plt.savefig('incomplete_'+eval[i] + '.png')

    plt.show()
    return


cross_validation()
