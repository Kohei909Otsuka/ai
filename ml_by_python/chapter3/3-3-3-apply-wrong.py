# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 合成データを作成
X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)
# 訓練セットとテストセットに分割 
X_train, X_test = train_test_split(X, random_state=5, test_size=0.1)

# original: 訓練セットとテストセットを描画
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].scatter(X_train[:, 0], X_train[:, 1],
  c=mglearn.cm2(0), label="Training set", s=60)
axes[0].scatter(X_test[:, 0], X_test[:, 1],
  c=mglearn.cm2(1), label="Test set", s=60)
axes[0].legend(loc="upper left")
axes[0].set_title("Original Data")


# MinMaxScalerでscale
minMaxScaler = MinMaxScaler()
minMaxScaler.fit(X_train)
X_train_scaled = minMaxScaler.transform(X_train)
X_test_scaled = minMaxScaler.transform(X_test)

# scaled in same way: 訓練セットとテストセットを描画
axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
  c=mglearn.cm2(0), label="Training set", s=60)
axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1],
  c=mglearn.cm2(1), label="Test set", s=60)
axes[1].legend(loc="upper left")
axes[1].set_title("Scaled correct Data")

# scaled in wrong way: 訓練セットとテストセットを描画
newScaler = MinMaxScaler()
newScaler.fit(X_test)
X_test_scaled_badly = newScaler.transform(X_test)
axes[2].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1],
  c=mglearn.cm2(0), label="Training set", s=60)
axes[2].scatter(X_test_scaled_badly[:, 0], X_test_scaled_badly[:, 1],
  c=mglearn.cm2(1), label="Test set", s=60)
axes[2].legend(loc="upper left")
axes[2].set_title("Scaled wrong Data")

plt.show()
