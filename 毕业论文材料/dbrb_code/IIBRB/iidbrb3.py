import time

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from IIBRB.base import *
from IIBRB.rule import Rule
from IIBRB.differential_evolution import run_DE_algorithm
import numpy as np
import pandas as pd


def similar_function(A, rule, alpha):
    """
    第j个输入对第k条规则的相似度
    :param A: 前提属性参考值
    :param rule:
    :param alpha: 输入数据的置信分布
    :return:
    """
    similarities = []
    for i in range(len(A)):
        for j in range(len(A[i])):
            if rule.condition[i] == A[i][j]:
                similarities.append(alpha[i][j])
                break
            if j == len(A[i]) - 1:
                raise RuntimeError('value don\'t match any references (#属性值没有匹配任何的属性候选值)')

    return similarities


def train_params_func(dbrb_model, X, y, NIND, MAXGEN):
    """
    参数训练方法
    :param dbrb_model:
    :param X:
    :param y:
    :return:
    """
    return run_DE_algorithm(dbrb_model, X, y, NIND, MAXGEN)


def transform_to_belief(X, A, delta):
    """
    :param X: 一维数据
    :param A:
    :return:
    """
    match_set = []
    for i in range(np.shape(A)[0]):
        match_degree = [0] * len(A[i])
        if pd.isnull(X[i]):
            match_degree = [np.nan] * np.shape(A)[1]  # 生成参考属性匹配度都是nan的list
        else:
            # for j in range(len(A[i]) - 1):
            #     if X[i] <= A[i][j + 1]:
            #         match_degree[j] = (A[i][j + 1] - X[i]) / (A[i][j + 1] - A[i][j])
            #         match_degree[j + 1] = 1 - match_degree[j]
            #         break
            for j in range(np.shape(A)[1] - 1):
                if A[i][j + 1] >= X[i]:
                    match_degree[j] = np.exp(-1 * (X[i] - A[i][j]) ** 2 / (2 * delta ** 2))
                    match_degree[j + 1] = np.exp(-1 * (A[i][j + 1] - X[i]) ** 2 / (2 * delta ** 2))
                    match_degree[j] /= (match_degree[j] + match_degree[j + 1])
                    match_degree[j + 1] = 1 - match_degree[j]
                    break
        match_set.append(match_degree)
    return match_set


def calcu_support_val(recoder):
    """
    计算规则支撑值
    """
    res = []
    for j in range(np.shape(recoder)[1]):
        sum, count = 0, 0
        for i in range(np.shape(recoder)[0]):
            if recoder[i][j] > EPS:
                sum += recoder[i][j]
                count += 1
        if count != 0:
            res.append((sum / count) * (count / len(recoder)))
        else:
            res.append(0)
    return res


class DBRBBase(BaseEstimator):
    def __init__(self, A, D, sigma=None, delta=1/8, is_classify=False):
        """
        :param A: 属性参考值
        :param D: 评价结果等级
        :param sigma: 属性权重
        """
        self.A = A
        self.D = D
        self.delta = delta  # 调节因子
        if sigma is None:
            self.sigma = [1.0] * len(A)
        else:
            self.sigma = sigma
        self.rules = None
        self.average_process_time = None

        self.similar = None
        self.antecedentBeliefBase = None
        self.is_classify = is_classify

    def set_params(self, params):
        """
        设置参数
        :param params: list: n_sigma属性权重 + k_theta规则权重 + m_beta规则后件置信度 + t_utility规则前件效用
        :return:
        """
        if len(params) < 1:
            raise RuntimeError("没有传入参数到BRB中")
        else:
            params = np.array(params)
            index = 0
            self.sigma = []
            for N in range(len(self.A)):
                self.sigma.append(params[index])
                index += 1

            for k in range(len(self.rules)):
                rule = self.rules[k]
                rule.theta = params[index]
                index += 1

            for M in range(len(self.rules)):
                rule = self.rules[M]
                rule.beta = params[index: index + len(self.D)]
                index += len(self.D)

    def get_params_num(self):
        """
        获取参数个数：属性权重 + 规则后件置信度 + 规则权重 + 规则前件效用
        :return:
        """
        return len(self.A) + len(self.D) * len(self.rules) + len(self.rules)

    def train_result(self):
        """
        训练方法中调用该方法进行预测
        :param n_samples:
        :return:
        """
        n_samples = len(self.similar)
        n_features = len(self.A)
        y = np.zeros(n_samples)

        for i in range(n_samples):
            active_weight = calc_active_weights(self.similar[i], self.rules, n_features, self.sigma)
            if active_weight is None:
                continue
            beta = evidence_reasoning(active_weight, self.rules, self.D)

            if not self.is_classify:
                y[i] = 0
                for j in range(len(beta)):
                    y[i] += self.D[j] * beta[j]
            else:
                y[i] = self.D[np.argmax(beta)]
        return y

    def resolve_incomplete(self, incomplete_data):
        """
        处理不完备数据集
        :param incomplete_data:  不完备的输入数据的隶属度分布
        :return: complete-data
        """
        # 寻找缺失值得索引
        miss_data_index = set()
        for k in range(len(incomplete_data)):
            if pd.isnull(incomplete_data[k][0]):
                miss_data_index.add(k)
                continue
        for index in miss_data_index:
            support_recoder = []
            for k in range(np.shape(self.antecedentBeliefBase)[1]):
                if k in miss_data_index:
                    continue
                recoder = []
                for j in range(np.shape(self.antecedentBeliefBase)[0]):
                    if self.all_is_the_same(j, k, incomplete_data):
                        recoder.append(self.antecedentBeliefBase[j][index])
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
            if self.antecedentBeliefBase[j][k][i] > EPS > incomplete_data[k][i] \
                    or self.antecedentBeliefBase[j][k][i] < EPS < incomplete_data[k][i]:
                return False
        return True

    def fit(self, X, y, NIND, MAXGEN):
        """
        :param X: array-like: [n_simples, n_features]
        :param y: list: [n_simples]
        :return:
        """
        pre_rules = generate_antecedent(self.A)
        rules = []
        for k in range(len(pre_rules)):
            rule = Rule(self.A, self.D)
            rule.condition = pre_rules[k]
            rules.append(rule)
        self.rules = rules

        # 保存输入数据与规则属性的匹配度
        self.similar = []
        self.antecedentBeliefBase = []
        for i in range(np.shape(X)[0]):
            alpha = transform_to_belief(X[i], self.A, self.delta)
            self.antecedentBeliefBase.append(alpha)  # 将完整数据的置信分布存入前件置信库中
            self.similar.append(calc_rule_match_degree(similar_function, self.A, self.rules, alpha))

        train_params_func(self, X, y, NIND, MAXGEN)  # 参数训练
        return self

    def predict(self, X):
        """
        训练后调用该方法进行预测
        :param X:
        :param is_classify: true:分类； False:回归
        :return:
        """
        n_samples = np.shape(X)[0]
        n_features = np.shape(X)[1]
        y = np.zeros(n_samples)

        time_start = time.time()

        for i in range(n_samples):
            alpha = transform_to_belief(X[i], self.A, self.delta)
            alpha = self.resolve_incomplete(alpha)  # 处理不完整输入数据，得到一个完备的前件置信分布
            similar = calc_rule_match_degree(similar_function, self.A, self.rules, alpha)
            active_weight = calc_active_weights(similar, self.rules, n_features, self.sigma)
            beta = evidence_reasoning(active_weight, self.rules, self.D)

            if not self.is_classify:
                y[i] = 0
                for j in range(len(beta)):
                    y[i] += self.D[j] * beta[j]
            else:
                y[i] = self.D[np.argmax(beta)]  # 输出最大置信度的分类

        time_end = time.time()
        if n_samples > 0:
            self.average_process_time = (time_end - time_start) / n_samples
        return y


class IIDBRBRegressor(DBRBBase, RegressorMixin):
    def __init__(self, A, D, delta=1/8, sigma=None):
        super().__init__(A, D, sigma, delta)

    def fit(self, X, y, NIND=100, MAXGEN=100):
        super().fit(X, y, NIND, MAXGEN)
        return self

    def predict(self, X):
        return super().predict(X)


class IIDBRBClassifier(DBRBBase, ClassifierMixin):
    def __init__(self, A, D, delta=1/8, sigma=None):
        super().__init__(A, D, sigma, delta, True)

    def fit(self, X, y, NIND=100, MAXGEN=100):
        super().fit(X, y, NIND, MAXGEN)
        return self

    def classify(self, X):
        return super().predict(X)
