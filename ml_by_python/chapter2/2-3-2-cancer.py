# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
# from IPython.display import display

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))
print("shape of cancer data: \n{}".format(cancer.data.shape))
print("良性・悪性のカウント \n{}".format(
  {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
print("特徴量の種類: \n{}".format(cancer.feature_names))

