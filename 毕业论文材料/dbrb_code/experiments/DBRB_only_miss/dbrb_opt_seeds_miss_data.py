import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from DBRB_opt.dbrb import DBRBClassifier

from datasets.load_data import load_seeds
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
    for i in range(np.shape(X)[1]):  # 设置缺失第i个属性值
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
    X, y = load_seeds()
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
    X, y = load_seeds()
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
# [0.7095238095238096, 0.6761904761904762, 0.7, 0.6238095238095238, 0.6285714285714286, 0.6047619047619047, 0.6238095238095237, 0.5904761904761905, 0.5476190476190477]
# [0.6019047619047618, 0.6338095238095238, 0.5914285714285714, 0.5676190476190476, 0.5676190476190476, 0.5576190476190476, 0.5514285714285714, 0.5352380952380952, 0.5057142857142859]
# === unify miss ===
best_acc = []
avg_acc = []
for i in range(9):
    best, avg = unify_miss_CV((i+1)/10)
    best_acc.append(best)
    avg_acc.append(avg)
print(best_acc)
print(avg_acc)
# [0.6448979591836734, 0.6414965986394557, 0.7054421768707482, 0.6619047619047619, 0.6306122448979591, 0.6517006802721088, 0.680952380952381, 0.6598639455782314, 0.6680272108843538]
# [0.6448979591836734, 0.6414965986394557, 0.7054421768707482, 0.6619047619047619, 0.6306122448979591, 0.6517006802721088, 0.680952380952381, 0.6598639455782314, 0.6680272108843538]