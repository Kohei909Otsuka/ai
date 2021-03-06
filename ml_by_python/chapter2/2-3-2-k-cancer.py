# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
# from IPython.display import display

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target,random_state=66)

training_accuracy = []
test_accuracy = []
neighbors_settings = range(1,11)

for neighbors in neighbors_settings:
  clf = KNeighborsClassifier(n_neighbors=neighbors)
  clf.fit(X_train, y_train)
  # 訓練データに対する精度
  training_accuracy.append(clf.score(X_train, y_train))
  # targetデータに対する精度
  test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label="training_accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test_accuracy")
plt.xlabel("neighbors")
plt.ylabel("accuracy")
plt.legend()
plt.show()
