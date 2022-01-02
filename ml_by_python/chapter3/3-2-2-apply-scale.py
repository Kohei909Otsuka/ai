# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target,random_state=66)

print("train shape: {}".format(X_train.shape))
print("test shape: {}".format(X_test.shape))

# trainを変換
minMaxScaler = MinMaxScaler()
minMaxScaler.fit(X_train)
X_train_scaled = minMaxScaler.transform(X_train)
print("transformed shape: {}".format(X_train_scaled.shape))
print("pre-feature min before:\n {}".format(X_train.min(axis=0)))
print("pre-feature max before:\n {}".format(X_train.max(axis=0)))
print("pre-feature min after:\n {}".format(X_train_scaled.min(axis=0)))
print("pre-feature max after:\n {}".format(X_train_scaled.max(axis=0)))

# targetも変換
X_test_scaled = minMaxScaler.transform(X_test)
print("pre-feature min after test data:\n {}".format(X_test_scaled.min(axis=0)))
print("pre-feature max after test data:\n {}".format(X_test_scaled.max(axis=0)))
