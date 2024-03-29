import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from EBRB.liu_ebrb import LiuEBRBClassifier

from datasets.load_data import load_mammographic
from datasets.process_data import process_to_pieces
import random
import pandas as pd
from miss_data_method.random_process import random_array


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
    A, D = process_to_pieces(X, y, 2)
    kf = KFold(n_splits=N_SPLITS, shuffle=True)
    ebrb = LiuEBRBClassifier(A, D)

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
                # 均值
                # for c in range(np.shape(X_copy)[1]):
                #     mi = np.nanmean(X_copy[:, c])
                #     for r in range(np.shape(X_copy)[0]):
                #         if pd.isnull(X_copy[r][c]):
                #             X_copy[r][c] = mi
                # 众数、中位数
                for c in range(np.shape(X_copy)[1]):
                    tmp = []
                    for t in X_copy[:, c]:
                        if not pd.isnull(t):
                            tmp.append(t)
                    number = np.median(tmp)
                    # number = np.argmax(np.bincount(tmp))
                    for r in range(np.shape(X_copy)[0]):
                        if pd.isnull(X_copy[r][c]):
                            X_copy[r][c] = number
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
    :return: 返回十折实验后的平均数据
    """
    N_SPLITS = 10
    A, D = process_to_pieces(X, y, 3)
    kf = KFold(n_splits=N_SPLITS, shuffle=True)
    ebrb = LiuEBRBClassifier(A, D)

    maes = []
    times = []
    for train_index, test_index in kf.split(X):
        train_X, train_y = X[train_index, :], y[train_index]
        test_X, test_y = X[test_index], y[test_index]
        X_copy = random_array(train_X, miss_percent)
        # 均值
        # for c in range(np.shape(X_copy)[1]):
        #     mi = np.nanmean(X_copy[:, c])
        #     for r in range(np.shape(X_copy)[0]):
        #         if pd.isnull(X_copy[r][c]):
        #             X_copy[r][c] = mi
        # 众数、中位数
        for c in range(np.shape(X_copy)[1]):
            tmp = []
            for t in X_copy[:, c]:
                if not pd.isnull(t):
                    tmp.append(t)
            # number = np.median(tmp)
            number = np.argmax(np.bincount(tmp))
            for r in range(np.shape(X_copy)[0]):
                if pd.isnull(X_copy[r][c]):
                    X_copy[r][c] = number
        ebrb = ebrb.fit(X_copy, train_y)
        y_predict = ebrb.predict(test_X)
        maes.append(accuracy_score(y_predict, test_y))
        times.append(ebrb.average_process_time)

    return np.mean(maes), np.std(maes), np.mean(times)


def unify_miss_CV(miss_percent):
    if miss_percent >= 1:
        raise RuntimeError("数据缺失率应该小于1")
    X, y = load_mammographic()
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
    X, y = load_mammographic()

    best_acc, best_std, best_time = 0, 0, 0
    avg_acc = 0
    for tc in range(20):
        print("-------------------我是分割线(%d)--------------------------" % (tc + 1))
        acc, std, times = random_miss(X, y, miss_percent)
        avg_acc = (tc * avg_acc + acc) / (tc + 1)
        if acc > best_acc:
            best_acc, best_std, best_time = acc, std, times
        print("acc:%f(std:%f), avg_process_time:%f" % (acc, std, times))
        print("")
        print("miss percent: %d%%" % (miss_percent * 100))
        print("acc:%f(std:%f), avg_process_time:%f" % (acc, std, times))
        print("best_acc:%f(std:%f), avg_process_time:%f" % (best_acc, best_std, best_time))
        print("avg_acc:%f" % avg_acc)
        print("")
    return best_acc, avg_acc


# === random miss ===
# best_acc, avg_acc = [], []
# for per in range(9):
#     best, avg = random_miss_CV((per + 1) / 10)
#     best_acc.append(best)
#     avg_acc.append(avg)
# print(best_acc)
# print(avg_acc)
# 众数 [0.7792771084337351, 0.7765662650602411, 0.7700602409638554, 0.7627108433734942, 0.758855421686747, 0.7492771084337349, 0.7387349397590363, 0.7162048192771086, 0.6568072289156627]
# 中位数[0.7775903614457833, 0.7803614457831326, 0.776144578313253, 0.772409638554217, 0.767590361445783, 0.7614457831325302, 0.7473493975903613, 0.7179518072289156, 0.6413253012048192]


# === single miss ===
best_acc = []
avg_acc = []
for i in range(9):
    best, avg = unify_miss_CV((i+1)/10)
    best_acc.append(best)
    avg_acc.append(avg)
print(best_acc)
print(avg_acc)
print('mediam')
# 中位数
