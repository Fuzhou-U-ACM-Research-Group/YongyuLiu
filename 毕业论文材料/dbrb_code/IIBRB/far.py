from IIBRB.iidbrb3 import transform_to_belief, calcu_support_val
from datasets.process_data import calc_attr_belief
import numpy as np
import pandas as pd
EPS = 1e-6


class FAR:
    def __init__(self, X):
        self.fuzzy_belief_express_base = []
        self.A = calc_attr_belief(X, 5)
        for i in range(np.shape(X)[0]):
            alpha = transform_to_belief(X[i], self.A, 1)
            self.fuzzy_belief_express_base.append(alpha)  # 将完整数据的置信分布存入前件置信库中

    def fill_in(self, incomplete_data_set):
        data = []
        for incomplete_data in incomplete_data_set:
            incomplete_data = transform_to_belief(incomplete_data, self.A, 1)
            self.resolve(incomplete_data)
            data.append(self.fusion(incomplete_data))
        return np.array(data)

    def resolve(self, incomplete_data):
        miss_data_index = set()
        for k in range(len(incomplete_data)):
            if pd.isnull(incomplete_data[k][0]):
                miss_data_index.add(k)
                continue
        for index in miss_data_index:
            support_recoder = []
            for k in range(np.shape(self.fuzzy_belief_express_base)[1]):
                if k in miss_data_index:
                    continue
                recoder = []
                for j in range(np.shape(self.fuzzy_belief_express_base)[0]):
                    if self.all_is_the_same(j, k, incomplete_data):
                        recoder.append(self.fuzzy_belief_express_base[j][index])
                if len(recoder) != 0:
                    support_recoder.append(calcu_support_val(recoder))
            if len(support_recoder) != 0:
                incomplete_data[index] = np.mean(support_recoder, axis=0)
            else:
                incomplete_data[index] = np.zeros(len(incomplete_data[index]))
        return incomplete_data

    def all_is_the_same(self, j, k, incomplete_data):
        """
        判断属性值是否一致
        """
        for i in range(len(incomplete_data[k])):
            if self.fuzzy_belief_express_base[j][k][i] > EPS > incomplete_data[k][i] \
                    or self.fuzzy_belief_express_base[j][k][i] < EPS < incomplete_data[k][i]:
                return False
        return True

    def fusion(self, incomplete):
        res = []
        for i in range(len(incomplete)):
            t = 0
            for j in range(len(incomplete[i])):
                t += self.A[i][j] * incomplete[i][j]
            res.append(t)
        return np.array(res)
