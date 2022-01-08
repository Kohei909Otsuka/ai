# -*- coding: utf-8 -*-
import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# page: 60
# 入力がx1, x2とb(3 node)
# が
# 3 nodeの隠れ層に接続されていることを考える
# 隠れそう層の出力の行列をAとすると
# A = XW + Bとなる
# X = [1.0, 0.5]
# W = [
#   [0.1, 0.3, 0.5]
#   [0.2, 0.4, 0.6]
# ]
# B  = [0.1, 0.2, 0.3]
# とすると...

X = np.array([1.0, 0.5])
# 1層目の重みW1
W1 = np.array([
  [0.1, 0.3, 0.5],
  [0.2, 0.4, 0.6],
])
# 1層目のバイアスB1
B1 = np.array([0.1, 0.2, 0.3])
print("X")
print(X)
print("W1")
print(W1)
print("B1")
print(B1)
print("1層目: 行列の積 X * W: {}".format(np.dot(X, W1)))
print("1層目: 行列の積とBの和: {}".format(np.dot(X, W1) + B1))

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)
print("1層目: sigmoid関数を重み付き和に対して適用: {}".format(Z1))
print("")
print("")

# 2層目の重みW2(入力が3つでnodeが2つ)
W2 = np.array([
  [0.1, 0.4],
  [0.2, 0.5],
  [0.3, 0.6],
])
# 2層目のバイアスB2
B2 = np.array([0.1, 0.2])
print("W2")
print(W2)
print("B2")
print(B2)
print("2層目: 行列の積 X * W: {}".format(np.dot(Z1, W2)))
print("2層目: 行列の積とBの和: {}".format(np.dot(Z1, W2) + B2))
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
print("2層目: sigmoid関数を重み付き和に対して適用: {}".format(Z2))
print("")
print("")
