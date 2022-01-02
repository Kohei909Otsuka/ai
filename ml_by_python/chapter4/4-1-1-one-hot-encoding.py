# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
import pandas as pd
import os


adult_path = os.path.join(mglearn.datasets.DATA_PATH, "adult.data")
data = pd.read_csv(
  adult_path,
  header=None,
  index_col=False,
  names=[
    'age',
    'workClass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'gender',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income'
  ]
)

# 解説のため、いくつかの特徴量だけに絞る
data = data[[
  'age',
  'workClass',
  'education',
  'gender',
  'hours-per-week',
  'occupation',
  'income'
]]

print(data.head())
print(data.gender.value_counts())
print(data.income.value_counts())
print(data.education.value_counts())

print("original features: \n", list(data.columns), "\n")
data_dummies = pd.get_dummies(data)
print("dumied features: \n", list(data_dummies.columns), "\n")
print(data_dummies.head())


features = data_dummies.loc[:, 'age':'occupation_ Transport-moving']
X = features.values
y = data_dummies['income_ >50K'].values
print("X.shape: {}, y.shape: {}".format(X.shape, y.shape))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logres = LogisticRegression()
logres.fit(X_train, y_train)
print("LogisticRegression test socre: {}".format(logres.score(X_test, y_test)))
