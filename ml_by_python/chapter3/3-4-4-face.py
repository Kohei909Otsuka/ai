# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)

print("shape of people: {}".format(people.images.shape))
print("shape of image: {}".format(people.images[0].shape))
print("shape of data: {}".format(people.data.shape))

# 画像を表示(なんかできない)
# fix, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': {}})
# for target, image, ax in zip(people.target, people.images, axes.ravel()):
#    ax.imshow(image)
#    ax.set_title(people.target_names[target])
# plt.show()

# 画像をカウント

counts = np.bincount(people.target)
for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print("{0:25} {1:3}".format(name, count))


mask = np.zeros(people.target.shape, dtype=np.bool)
print("people.target.shape: {}".format(people.target.shape))
print("mask: {}".format(mask))
print("people.target: {}".format(people.target))

for target in np.unique(people.target):
  mask[np.where(people.target == target)[0][:50]] = 1

print("mask after: {}".format(mask))

X_people = people.data[mask]
y_people = people.target[mask]

# 0 ~ 255で表現される白黒度合いを0~1に変換
X_people = X_people / 255.0

# originalのピクセルの白黒の度合いをそのまま使うと精度が23%とかそのあたり
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("knnのscore: {}".format(knn.score(X_test, y_test)))

# mglearn.plots.plot_pca_whitening()
# plt.show()

# PCAで変換してから解析する30%に精度があがった
pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("X_train_pca.shape: {}".format(X_train_pca.shape))
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_pca, y_train)
print("pca込knnのscore: {}".format(knn.score(X_test_pca, y_test)))
