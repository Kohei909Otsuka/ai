# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

X, y = mglearn.datasets.make_wave(n_samples=100)

line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
reg = DecisionTreeRegressor(min_samples_split=3).fit(X, y)
plt.plot(line, reg.predict(line), label="decission tree")

reg = LinearRegression().fit(X, y)
plt.plot(line, reg.predict(line), label="linear")

plt.plot(X[:, 0], y, 'o', c='k')
plt.xlabel("Input feature")
plt.ylabel("regression output")
plt.legend(loc="best")
# plt.show()

# Xを11個のビンにわける
bins = np.linspace(-3, 3 ,11)
print("bins: {}".format(bins))
which_bin = np.digitize(X, bins=bins)
print("first five data: \n{}".format(X[:5]))
print("first five membership: \n{}".format(which_bin[:5]))

# ビンのままではenumみたいなものなので、扱えるようにone hot encodingする
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
encoder.fit(which_bin)
X_binned = encoder.transform(which_bin)
print("one hot encoded X: \n{}".format(X_binned[:5]))

# ビン分けされたものでもう一度可視化
line_binned = encoder.transform(np.digitize(line, bins=bins))

reg = LinearRegression().fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label="linear binned")

reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label="decission tree binned")

plt.plot(X[:, 0], y, 'o', c='k')
plt.xlabel("Input feature")
plt.ylabel("regression output")
plt.legend(loc="best")
plt.show()
