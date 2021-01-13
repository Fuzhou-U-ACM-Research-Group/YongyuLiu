from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
import numpy as np


def ebrb_model(ebrb_model, train_X, train_y, test_X, test_y):
    """
    EBRB模型
    :param ebrb_model:
    :param train_X:
    :param train_y:
    :param test_X:
    :param test_y:
    :return:
    """
    ebrb_model.fit(train_X, train_y)
    y_predict = ebrb_model.predict(test_X)
    y_1 = []
    y_2 = []
    for index in range(len(y_predict)):  # 去掉零激活
        if y_predict[index] == 0:
            continue
        y_1.append(y_predict[index])
        y_2.append(test_y[index])
    return [mean_absolute_error(y_2, y_1), np.sqrt(mean_squared_error(y_2, y_1)), r2_score(y_2, y_1),
            r2_score_adjusted(y_2, y_1, np.shape(test_X)[1]), explained_variance_score(y_2, y_1)]


def brb_model(brb, train_X, train_y, test_X, test_y):
    brb.fit(train_X, train_y)
    y_predict = brb.predict(test_X)
    return [mean_absolute_error(test_y, y_predict), np.sqrt(mean_squared_error(test_y, y_predict)),
            r2_score(test_y, y_predict),
            r2_score_adjusted(test_y, y_predict, np.shape(test_X)[1]), explained_variance_score(test_y, y_predict)]


def dbrb_model(dbrb, train_X, train_y, test_X, test_y):
    dbrb.fit(train_X, train_y)
    y_predict = dbrb.predict(test_X)
    return [mean_absolute_error(test_y, y_predict), np.sqrt(mean_squared_error(test_y, y_predict)),
            r2_score(test_y, y_predict),
            r2_score_adjusted(test_y, y_predict, np.shape(test_X)[1]), explained_variance_score(test_y, y_predict)]


def svr_regression_model(svr_model, train_X, train_y, test_X, test_y):
    """
    支持向量回归模型
    :param svr_model:
    :param train_X:
    :param train_y:
    :param test_X:
    :param test_y:
    :return:
    """
    svr_model = svr_model
    svr_model.fit(train_X, train_y)
    y_predict = svr_model.predict(test_X)
    return [mean_absolute_error(test_y, y_predict), np.sqrt(mean_squared_error(test_y, y_predict)), r2_score(test_y, y_predict),
            r2_score_adjusted(test_y, y_predict, np.shape(test_X)[1]), explained_variance_score(test_y, y_predict)]


def decision_tree_model(decision_tree_model, train_X, train_y, test_X, test_y, indexList=None):
    """
    决策树模型
    :param decision_tree_model: 决策树模型
    :param indexList: 需要离散化的列值
    :param train_X: 训练数据
    :param train_y: 训练结果
    :param test_X: 预测数据
    :param test_y: 预测结果
    """
    if indexList is not None:
        for row in range(len(indexList)):
            '''
            使用简单的二分法
            '''
            col = indexList[row]
            col_data = train_X[:, col]
            col_data = np.reshape(col_data, -1)
            median = np.median(col_data)
            for j in range(len(col_data)):
                train_X[j][col] = 0 if train_X[j][col] <= median else 1
            for j in range(len(test_X)):
                test_X[j][col] = 0 if test_X[j][col] <= median else 1

    model = decision_tree_model
    model.fit(train_X, train_y)
    y_predict = model.predict(test_X)

    return [mean_absolute_error(test_y, y_predict), np.sqrt(mean_squared_error(test_y, y_predict)),
            r2_score(test_y, y_predict),
            r2_score_adjusted(test_y, y_predict, np.shape(test_X)[1]), explained_variance_score(test_y, y_predict)]


def mlp_model(mlp, train_X, train_y, test_X, test_y):
    """
    多层感知器模型
    :param mlp:
    :param train_X:
    :param train_y:
    :param test_X:
    :param test_y:
    :return:
    """
    mlp.fit(train_X, train_y)
    y_predict = mlp.predict(test_X)
    return [mean_absolute_error(test_y, y_predict), np.sqrt(mean_squared_error(test_y, y_predict)),
            r2_score(test_y, y_predict),
            r2_score_adjusted(test_y, y_predict, np.shape(test_X)[1]), explained_variance_score(test_y, y_predict)]


def bys_model(bys, train_X, train_y, test_X, test_y):
    """
    贝叶斯模型
    :param bys:
    :param train_X:
    :param train_y:
    :param test_X:
    :param test_y:
    :return:
    """
    bys.fit(train_X, train_y)
    y_predict = bys.predict(test_X)
    return [mean_absolute_error(test_y, y_predict), np.sqrt(mean_squared_error(test_y, y_predict)),
            r2_score(test_y, y_predict),
            r2_score_adjusted(test_y, y_predict, np.shape(test_X)[1]), explained_variance_score(test_y, y_predict)]


def r2_score_adjusted(test_y, predict_y, features):
    n = len(test_y)
    p = features
    return 1 - (1 - r2_score(test_y, predict_y)) * (n - 1) / (n - p - 1)
