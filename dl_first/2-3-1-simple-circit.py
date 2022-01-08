# -*- coding: utf-8 -*-
import numpy as np

# yについて
# 0: w1w1 + w2w2 <= theta
# 1: w1w1 + w2w2 > theta
# と解釈した実装
def SIMPLE_AND(x1, x2):
  w1, w2, theta = 0.5, 0.5, 0.7
  tmp = (x1 * w1 ) + (x2 * w2)
  if tmp <= theta:
    return 0
  else:
    return 1

# thetaをbにかえて正負逆の意味にして(b == -theta)、右辺から左辺にもっていき
# 0: b + w1w1 + w2w2 <= 0
# 1: b + w1w1 + w2w2 > 0
# と解釈した実装
# 行列演算を使った実装
def AND(x1, x2):
  x = np.array([x1, x2])
  w = np.array([0.5, 0.5])
  b = -0.7
  tmp = np.sum(w * x) + b
  if tmp <= 0:
    return 0
  else:
    return 1

def NAND(x1, x2):
  x = np.array([x1, x2])
  w = np.array([-0.5, -0.5])
  b = 0.7
  tmp = np.sum(w * x) + b
  if tmp <= 0:
    return 0
  else:
    return 1

def OR(x1, x2):
  x = np.array([x1, x2])
  w = np.array([0.5, 0.5])
  b = -0.2
  tmp = np.sum(w * x) + b
  if tmp <= 0:
    return 0
  else:
    return 1

def XOR(x1, x2):
  s1 = NAND(x1, x2)
  s2 = OR(x1, x2)
  return AND(s1, s2)

print("AND回路の出力")
print("AND(0, 0): {}".format(AND(0,0)))
print("AND(0, 1): {}".format(AND(0,1)))
print("AND(1, 0): {}".format(AND(1,0)))
print("AND(1, 1): {}".format(AND(1,1)))

print("NAND回路の出力")
print("NAND(0, 0): {}".format(NAND(0,0)))
print("NAND(0, 1): {}".format(NAND(0,1)))
print("NAND(1, 0): {}".format(NAND(1,0)))
print("NAND(1, 1): {}".format(NAND(1,1)))

print("OR回路の出力")
print("OR(0, 0): {}".format(OR(0,0)))
print("OR(0, 1): {}".format(OR(0,1)))
print("OR(1, 0): {}".format(OR(1,0)))
print("OR(1, 1): {}".format(OR(1,1)))

print("XOR回路の出力")
print("XOR(0, 0): {}".format(XOR(0,0)))
print("XOR(0, 1): {}".format(XOR(0,1)))
print("XOR(1, 0): {}".format(XOR(1,0)))
print("XOR(1, 1): {}".format(XOR(1,1)))
