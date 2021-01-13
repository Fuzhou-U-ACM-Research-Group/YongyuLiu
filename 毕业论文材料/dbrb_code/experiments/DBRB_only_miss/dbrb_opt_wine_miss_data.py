import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from DBRB_opt.dbrb import DBRBClassifier

from datasets.load_data import load_wine
from datasets.process_data import process_to_pieces
import random
from miss_data_method.random_process import random_array
# import missingno
import pandas as pd


def unify_single_miss(X, y, miss_percent):
    """
    统一缺失单个属性：
        数据集X中只缺失某一个属性值，缺失的位置是随机的
    :param X:
    :param y:
    :param miss_percent: 训练数据集的缺失率
    :return: 返回list格式，其中第i个位置代表缺失第i个属性的平均数据
    """
    N_SPLITS = 10
    A, D = process_to_pieces(X, y, 3)
    kf = KFold(n_splits=N_SPLITS, shuffle=True)
    ebrb = DBRBClassifier(A, D)

    total_maes = []
    total_std = []
    total_times = []
    for i in range(1):  # 设置缺失第1个属性值
    # for i in range(np.shape(X)[1]):  # 设置缺失第i个属性值
        maes = []
        times = []
        for train_index, test_index in kf.split(X):
            train_X, train_y = X[train_index, :], y[train_index]
            test_X, test_y = X[test_index], y[test_index]
            X_copy = np.copy(train_X)
            for j in random.sample([value for value in range(len(train_X))],
                                   int(miss_percent * len(train_X))):  # 随机获取空值索引
                X_copy[j][i] = None  # 将获取的索引位置设置为空
            # missingno.matrix(pd.DataFrame(X_copy))
            ebrb = ebrb.fit(X_copy, train_y)
            y_predict = ebrb.predict(test_X)
            maes.append(accuracy_score(y_predict, test_y))
            times.append(ebrb.average_process_time)
        total_maes.append(np.mean(maes))
        total_std.append(np.std(maes))
        total_times.append(np.mean(times))
    return total_maes, total_std, total_times


def random_miss(X, y, miss_percent):
    """
    随机缺失多个属性值
    :param X:
    :param y:
    :param miss_percent: 训练数据集的缺失率
    :param miss_feature_nums: 缺失属性个数
    :return: 返回十折实验后的平均数据
    """
    N_SPLITS = 10
    A, D = process_to_pieces(X, y, 3)
    kf = KFold(n_splits=N_SPLITS, shuffle=True)
    ebrb = DBRBClassifier(A, D)

    maes = []
    times = []
    for train_index, test_index in kf.split(X):
        train_X, train_y = X[train_index, :], y[train_index]
        test_X, test_y = X[test_index], y[test_index]
        X_copy = random_array(train_X, miss_percent)
        # missingno.matrix(pd.DataFrame(X_copy)) # 获取缺失数据分布情况图
        ebrb = ebrb.fit(X_copy, train_y)
        y_predict = ebrb.predict(test_X)
        maes.append(accuracy_score(y_predict, test_y))
        times.append(ebrb.average_process_time)

    return np.mean(maes), np.std(maes), np.mean(times)


def unify_miss_CV(miss_percent):
    if miss_percent >= 1:
        raise RuntimeError("数据缺失率应该小于1")
    X, y = load_wine()
    total_acc = []
    total_std = []
    total_times = []
    best_acc, mean = 0, 0
    for tc in range(1):
        print("-------------------我是分割线(%d)--------------------------" % (tc + 1))
        acc, std, times = unify_single_miss(X, y, miss_percent)
        total_acc.append(acc)
        total_std.append(std)
        total_times.append(times)
        print("acc: %s " % acc)
        print("std: %s" % std)
        print("avg_process_time: %s" % times)
        print("")
        best_acc = np.max(total_acc, 0)
        best_idx = np.argmax(total_acc, 0)
        idx = [i for i in range(len(acc))]
        print("miss data percent: %f%%" % int(miss_percent * 100))
        print("best_acc: %s(avg: %s)" % (best_acc, np.mean(best_acc)))
        print("std: %s" % np.array(total_std)[best_idx, idx])
        print("time: %s" % np.array(total_times)[best_idx, idx])
        mean = np.mean(total_acc, 0)
        print("avg_acc: %s(avg: %s)" % (mean, np.mean(mean)))
        print("")
    return np.mean(best_acc), np.mean(mean)


def random_miss_CV(miss_percent):
    if miss_percent >= 1:
        raise RuntimeError("数据缺失率应该小于1")
    X, y = load_wine()
    best_acc, best_std, best_time = 0, 0, 0
    avg_acc = 0
    for tc in range(10):
        print("-------------------我是分割线(%d)--------------------------" % (tc + 1))
        acc, std, times = random_miss(X, y, miss_percent)
        avg_acc = (tc * avg_acc + acc) / (tc + 1)
        if acc > best_acc:
            best_acc, best_std, best_time = acc, std, times
        print("acc:%f(std:%f), avg_process_time:%f" % (acc, std, times))
        print("")
        print("miss data percent: %f%%" % int(miss_percent * 100))
        print("best_acc:%f(std:%f), avg_process_time:%f" % (best_acc, best_std, best_time))
        print("avg_acc:%f" % avg_acc)
        print("")
    return best_acc, avg_acc


# === random miss ==
# best_acc, avg_acc = [], []
# for per in range(9):
#     best, avg = random_miss_CV((per + 1) / 10)
#     best_acc.append(best)
#     avg_acc.append(avg)
# print(best_acc)
# print(avg_acc)
# [0.7290849673202614, 0.6803921568627451, 0.719281045751634, 0.6911764705882353, 0.6130718954248366, 0.5784313725490197, 0.5320261437908497, 0.4833333333333333, 0.45784313725490194]
# [0.6572549019607845, 0.6432352941176471, 0.6271241830065359, 0.6130718954248365, 0.5765359477124183, 0.5395751633986927, 0.48431372549019613, 0.4574509803921568, 0.4326470588235295]

# === unify miss ===
best_acc = []
avg_acc = []
for i in range(9):
    best, avg = unify_miss_CV((i+1)/10)
    best_acc.append(best)
    avg_acc.append(avg)
print(best_acc)
print(avg_acc)
# [0.6447712418300654, 0.6741830065359478, 0.6627450980392157, 0.6584967320261438, 0.6277777777777778, 0.5888888888888889, 0.6388888888888888, 0.6787581699346406, 0.5565359477124183]