from datasets.load_data import load_x_data_by_name
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from experiments.teachingEvaluation.set_pipline import data_cleansing


def attribute(data):
    X, y = data[:, :-1], data[:, -1]
    X, y = data_cleansing(X, y)
    pca = PCA(n_components='mle')
    # kf = KFold(n_splits=10, shuffle=True)
    # for train_index, test_index in kf.split(X):
    #     train_X, train_y = X[train_index, :], y[train_index]
    #     test_X, test_y = X[test_index], y[test_index]
    #     pca.fit(train_X)
    pca.fit(X)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)
    print(pca.components_)
    print(pca.mean_)


# data1 = load_x_data_by_name("jscp_201801_notAll_data_set.csv")
data2 = load_x_data_by_name("jscp_201801_all_data_set.csv")
# data = np.concatenate((data1, data2), axis=0)
attribute(data2)
