import time

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from DBRB_opt.base import *
from DBRB_opt.rule import Rule
from DBRB_opt.differential_evolution import run_DE_algorithm
import numpy as np

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
            # if j == len(A[i])-1:
            #     print(j)
            #     print(len(A[i]))
            #     raise RuntimeError('value don\'t match any references (#属性值没有匹配任何的属性候选值)')

    return similarities


def train_params_func(dbrb_model, X, y):
    """
    参数训练方法
    :param dbrb_model:
    :param X:
    :param y:
    :return:
    """
    return run_DE_algorithm(dbrb_model, X, y, 20, 50)


def transform_to_belief(X, A):
    """
    将输入数据转化为置信分布形式
    注：在参数优化过程中，可能会出现同个属性中两个参考值相同的情况
    :param X: 一维数据
    :param A:
    :return:
    """
    match_set = []
    for i in range(len(A)):
        match_degree = [0] * len(A[i])
        pre_j = 0
        for j in range(len(A[i]) - 1):
            if A[i][j] == A[i][j + 1]:
                continue
            if X[i] <= A[i][j + 1]:
                match_degree[j] = (A[i][j + 1] - X[i]) / (A[i][j + 1] - A[i][pre_j])
                match_degree[j + 1] = 1 - match_degree[j]
                break
            pre_j = j + 1

        match_set.append(match_degree)
    return match_set


class DBRBBase(BaseEstimator):
    def __init__(self, A, D, sigma=None, is_classify=False):
        """
        :param A: 属性参考值
        :param D: 评价结果等级
        :param sigma: 属性权重
        """
        self.A = A
        self.D = D
        if sigma is None:
            self.sigma = [1.0] * len(A)
        else:
            self.sigma = sigma
        self.rules = None
        self.average_process_time = None

        self.is_classify = is_classify

    def partition_params(self, params):
        """
        划分参数训练后的数据
        :param params: list  n_sigma属性权重 + k_theta规则权重 + m_beta规则后件置信度
        :return: res = {属性权重, 规则权重, 规则分布形式, 规则}
        """
        if len(params) < 1:
            raise RuntimeError('没有传入参数到BRB中')

        params = np.array(params)
        index = 0
        res = {'attr_weight': [], 'rule_weight': [], 'rules': [], 'rules_formation': []}
        # 获取属性权重
        for N in range(len(self.A)):
            res['attr_weight'].append(params[index])
            index += 1

        # 获取规则权重
        for k in range(len(self.rules)):
            res['rule_weight'].append(params[index])
            index += 1

        # 获取规则库
        for k in range(len(self.rules)):
            rule = Rule(self.A, self.D)
            rule.theta = res['rule_weight'][k]
            rule.condition = self.rules[k].condition
            rule.beta = params[index: index + len(self.D)]
            res['rules'].append(rule)
            res['rules_formation'].append(str(rule.condition) + ' ' + str(rule.beta))
            index += len(self.D)
        return res

    def set_params(self, params):
        """
        设置参数
        :param params: 系统参数
        :return:
        """
        part_params = self.partition_params(params)
        self.sigma = part_params['attr_weight']
        self.rules = part_params['rules']

    def get_params_num(self):
        """
        获取参数个数：属性权重 +  规则权重 + 规则后件置信度
        :return: {属性权重个数，规则权重个数，前件效用个数，后件置信度个数，总个数}
        """
        attr_weight_num = len(self.A)
        rule_weight_num = len(self.rules)
        belief_num = len(self.D) * len(self.rules)
        total = attr_weight_num + rule_weight_num + belief_num
        count = {'attr_weight_num': attr_weight_num, 'rule_weight_num': rule_weight_num,
                 'belief_num': belief_num, 'total': total}
        return count

    def fit(self, X, y):
        """
        :param X: array-like: [n_simples, n_features]
        :param y: list: [n_simples]
        :return:
        """
        pre_rules = generate_antecedent(self.A)  # 线性组合前件效用，生成规则
        rules = []
        for k in range(len(pre_rules)):
            rule = Rule(self.A, self.D)
            rule.condition = pre_rules[k]
            rules.append(rule)
        self.rules = rules

        train_params_func(self, X, y)  # 参数训练
        return self

    def predict(self, X):
        """
        训练后调用该方法进行预测
        :param X:
        is_classify: true:分类； False:回归
        :return:
        """
        n_samples = np.shape(X)[0]
        n_features = np.shape(X)[1]
        y = np.zeros(n_samples)

        self.sigma = [value / np.max(self.sigma) for value in self.sigma]  # average of sigma

        time_start = time.time()
        for i in range(n_samples):
            alpha = transform_to_belief(X[i], self.A)
            similar = calc_rule_match_degree(similar_function, self.A, self.rules, alpha)
            active_weight = calc_active_weights(similar, self.rules, n_features, self.sigma)
            if active_weight is None:
                # print('出现零激活')
                continue

            beta = evidence_reasoning(active_weight, self.rules, self.D)
            if not self.is_classify:
                y[i] = 0
                for j in range(len(beta)):
                    y[i] += self.D[j] * beta[j]

            else:
                y[i] = self.D[np.argmax(beta)]

        time_end = time.time()
        if n_samples > 0:
            self.average_process_time = (time_end - time_start) / n_samples
        return y


class DBRBRegressor(DBRBBase, RegressorMixin):
    def __init__(self, A, D, sigma=None):
        super().__init__(A, D, sigma)

    def fit(self, X, y):
        super().fit(X, y)
        return self

    def predict(self, X):
        return super().predict(X)


class DBRBClassifier(DBRBBase, ClassifierMixin):
    def __init__(self, A, D, sigma=None):
        super().__init__(A, D, sigma, True)

    def fit(self, X, y):
        super().fit(X, y)
        return self

    def classify(self, X):
        return super().predict(X)
