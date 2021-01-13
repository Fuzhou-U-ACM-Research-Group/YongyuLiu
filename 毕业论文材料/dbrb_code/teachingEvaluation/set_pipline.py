
from datasets.load_data import *
from teachingEvaluation.er import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# precondition = load_x_data_by_name('jscp_201801_pre_semester_all_precondition.csv')
# belief = load_x_data_by_name('jscp_201801_all_utility.csv')

# print('++++++++++++++++++++++++')
# data_set = []
# for ii in range(len(precondition)):
#     for j in range(len(belief)):
#         if precondition[ii][-2] == belief[j][-2] and precondition[ii][-1] == belief[j][-1]:
#             print(precondition[ii])
#             print(belief[j])
#             print("-----------------")
#             data_1 = [x for x in precondition[ii][:-2]]
#             data_2 = [x for x in belief[j][:-2]]
#             data = np.concatenate((data_1, data_2), axis=0)
#             data_set.append(data)
# write_csv_file_by_name("jscp_201801_pre_semester_data_set.csv", data_set)
from datasets.process_data import calc_attr_belief


def process_to_pieces(X, y, x_pieces=5):
    A = []
    for col in range(np.shape(X)[1]):
        if col == 5:  # 修习类型
            d = X[:, col]
            A.append(list(set(d)))
            continue
        if col == 2:  # 性别
            d = X[:, col]
            A.append(list(set(d)))
            continue
        col_min = np.min(X[:, col], axis=0)
        col_max = np.max(X[:, col], axis=0)
        A.append(list(np.linspace(col_min, col_max, x_pieces)))
    D = calc_attr_belief(y.reshape(-1, 1), 5)[0]
    return A, D


def data_cleansing(X, y):
    output_X = []
    output_y = []
    for i in range(np.shape(X)[0]):
        row = []
        flag = False
        for j in range(np.shape(X)[1]):
            if isinstance(X[i][j], str):
                if X[i][j] == '_' or X[i][j] == '___':
                    flag = True
                    break
                if X[i][j] == 'nan':
                    flag = True
                    break
                row.append(int(X[i][j]))
                continue
            elif np.isnan(X[i][j]):
                flag = True
                break
            row.append(X[i][j])
        if not flag:
            output_X.append(row)
            output_y.append(y[i])

    # print(len(X))
    # print(len(output_X))

    return np.array(output_X), np.array(output_y)


def data_cleansing_nan(X, y):
    output_X = []
    output_y = []
    for i in range(np.shape(X)[0]):
        row = []
        for j in range(np.shape(X)[1]):
            if isinstance(X[i][j], str):
                if X[i][j] == '_' or X[i][j] == '___' or X[i][j] == 'nan':
                    X[i][j] = np.nan
                    row.append(X[i][j])
                    continue
                row.append(int(X[i][j]))
                continue
            row.append(X[i][j])
        output_X.append(row)
        output_y.append(y[i])

    return np.array(output_X), np.array(output_y)
