# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
# from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 2.3.2 k-近傍
# mglearn.plots.plot_knn_classification(n_neighbors=1)
# mglearn.plots.plot_knn_classification(n_neighbors=3)
# plt.show()

X, y = mglearn.datasets.make_forge()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train) 
print("Test set prediction: {}".format(clf.predict(X_test)))
print("精度: {}".format(clf.score(X_test, y_test)))


# n_neighborsのパラメータによる変化の可視化
fig ,axes = plt.subplots(1, 3, figsize=(10, 3))

print(zip([1,3,9], axes))
for n_neighbors, ax in zip([1,3,9], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbors".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
plt.show()

