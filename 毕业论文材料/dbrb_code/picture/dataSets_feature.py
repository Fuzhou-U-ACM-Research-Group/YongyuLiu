from datasets.load_data import load_seeds
from datasets.load_data import load_iris
from datasets.load_data import load_wine
from datasets.load_data import load_transfusion
from datasets.load_data import load_mammographic
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


tsne = TSNE(n_components=2, init='pca', random_state=1)
#
X, y = load_iris()
result = tsne.fit_transform(X)
x_min, x_max = np.min(result), np.max(result)
result = (result - x_min)/(x_max-x_min)
ax = plt.subplot(231)
for i in range(result.shape[0]):
    plt.text(result[i,0], result[i,1], str(y[i]), color=plt.cm.Set1(y[i]), fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.title('iris')

#
X, y = load_seeds()
result = tsne.fit_transform(X)
x_min, x_max = np.min(result), np.max(result)
result = (result - x_min)/(x_max-x_min)
ax = plt.subplot(232)
for i in range(result.shape[0]):
    plt.text(result[i,0], result[i,1], str(y[i]), color=plt.cm.Set1(y[i]), fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.title('seeds')

#
X, y = load_wine()
result = tsne.fit_transform(X)
x_min, x_max = np.min(result), np.max(result)
result = (result - x_min)/(x_max-x_min)
ax = plt.subplot(233)
for i in range(result.shape[0]):
    plt.text(result[i,0], result[i,1], str(y[i]), color=plt.cm.Set1(y[i]), fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.title('wine')

#
X, y = load_transfusion()
x_min, x_max = np.min(result), np.max(result)
result = (result - x_min)/(x_max-x_min)
ax = plt.subplot(223)
for i in range(result.shape[0]):
    plt.text(result[i,0], result[i,1], str(y[i]), color=plt.cm.Set1(y[i]), fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.title('transfusion')

#
X, y = load_mammographic()
x_min, x_max = np.min(result), np.max(result)
result = (result - x_min)/(x_max-x_min)
ax = plt.subplot(224)
for i in range(result.shape[0]):
    plt.text(result[i,0], result[i,1], str(y[i]), color=plt.cm.Set1(y[i]), fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.title('mammographic')


plt.show()


