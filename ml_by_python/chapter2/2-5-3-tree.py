# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target,random_state=42)

# 決定木の制限なし
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("max 訓練精度: {}".format(tree.score(X_train, y_train)))
print("max 予測精度: {}".format(tree.score(X_test, y_test)))

# 決定木の制限あり
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
print("depth=4 訓練精度: {}".format(tree.score(X_train, y_train)))
print("depth=4 予測精度: {}".format(tree.score(X_test, y_test)))

# 木の可視化
export_graphviz(tree, out_file="tree.dot", class_names=["malignant", "benign"],
  feature_names=cancer.feature_names, impurity=False, filled=True)
# with open("tree.dot") as f:
  # dot_graph = f.read()
# graphviz.Source(dot_graph)
graphviz.render("dot", "png", "tree.dot")
