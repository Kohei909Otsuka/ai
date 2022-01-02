# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
# from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

# linerの表示
# mglearn.plots.plot_linear_regression_wave()
# plt.show()

# 最小二乗法(特徴量のすくないwave)

X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

Lr = LinearRegression().fit(X_train, y_train)
print("Lr.coef_: {}".format(Lr.coef_))
print("Lr.intercept_: {}".format(Lr.intercept_))
print("wave train score: {}".format(Lr.score(X_train, y_train)))
print("wave test score: {}".format(Lr.score(X_test, y_test)))
print("")

# 最小二乗法(特徴量の多いhousing)
X, y = mglearn.datasets.load_extended_boston()
print("house shape: {}".format(X.shape))
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
Lr = LinearRegression().fit(X_train, y_train)
print("house train score: {}".format(Lr.score(X_train, y_train)))
print("house test score: {}".format(Lr.score(X_test, y_test)))
print("")

# リッジ回帰
# 最小二乗法に「なるべく傾きを小さくする」という制限を加えたもの
X, y = mglearn.datasets.load_extended_boston()
print("house shape: {}".format(X.shape))
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
ridge = Ridge().fit(X_train, y_train)
print("house ridge train score: {}".format(ridge.score(X_train, y_train)))
print("house ridge test score: {}".format(ridge.score(X_test, y_test)))
print("")

# リッジ回帰のparams alphaを変えてみる(デフォルトは1.0で大きいほど傾きを小さくする力が増える)
# ridge = Ridge(alpha=1).fit(X_train, y_train)
# ridge10 = Ridge(alpha=10).fit(X_train, y_train)
# ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
# Lr = LinearRegression().fit(X_train, y_train)
# plt.plot(ridge01.coef_, 's', label="Ridge alpha=1")
# plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
# plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
# plt.plot(Lr.coef_, 'o', label="Linear")
# 
# plt.xlabel("_coef index")
# plt.ylabel("_coef")
# plt.hlines(0, 0, len(Lr.coef_))
# plt.ylim(-25, 25)
# plt.legend()
# plt.show()

# リッジ回帰のsample数を変えてみる
mglearn.plots.plot_ridge_n_samples()
plt.show()
