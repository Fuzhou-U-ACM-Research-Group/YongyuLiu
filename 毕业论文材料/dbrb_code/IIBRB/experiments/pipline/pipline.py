from IIBRB.iidbrb3 import IIDBRBRegressor
from datasets.load_data import load_data_by
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import math
import numpy as np

# oil_sizes = [500, 800, 1000, 1200, 1500, 2000]
oil_sizes = [500]
kase = 1
for oil_size in oil_sizes:
    A = [[-10, -5, -3, -1, 0, 1, 2, 3],
         [-0.042, -0.025, -0.01, 0.000, 0.01, 0.025, 0.042]]
    D = [0, 2, 4, 6, 8]  # 结果等级效用值
    sigma = [1.0, 0.001]  # 属性权重
    dbrb = IIDBRBRegressor(A, D, sigma)
    train_X, train_y = load_data_by("oil_traindata_%d.txt" % oil_size, path='data/oil-pip-line', sep=' ')
    dbrb = dbrb.fit(train_X, train_y)
    test_X, test_y = load_data_by("oil_testdata_2007.txt", path='data/oil-pip-line', sep=' ')
    y = dbrb.predict(test_X)

    print("----------------------我是分割线%d------------------" % kase)
    print("mae:", mean_absolute_error(y, test_y))
    # print("average_time:%f"%dbrb.average_process_time)
    kase += 1

# # 二维输出
l1 = plt.plot(range(len(test_X)), y, 'b', label='Liu-EBRB', linewidth=3)
l2 = plt.plot(range(len(test_X)), test_y, 'r--', label='real-values', linewidth=3)
plt.legend(loc='upper left', fontsize=30)
plt.show()
