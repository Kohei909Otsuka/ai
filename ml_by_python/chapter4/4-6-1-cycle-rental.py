# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
import locale

from sklearn.ensemble import RandomForestRegressor

# これがないと曜日が日本語になってグラフ描画できない
locale.setlocale(locale.LC_TIME, 'C.UTF-8')

citibike = mglearn.datasets.load_citibike()
print("citi bike data: \n{}".format(citibike.head()))

# 生データの可視化
# plt.figure(figsize=(10, 3))
xticks = pd.date_range(start=citibike.index.min(), end=citibike.index.max(), freq='D')
# plt.xticks(xticks, xticks.strftime("%a %m-%d"), rotation=90, ha="left")
# plt.plot(citibike, linewidth=1)
# plt.xlabel("Date")
# plt.ylabel("Rentals")
# plt.show()


# ターゲット値(レンタル数)
y = citibike.values
# UNIX時間にしたものを入力とする
X = citibike.index.astype("int64").to_numpy().reshape(-1,1) // 10**9

# 最初の184個のデータを訓練に、残りをテストに
n_train = 184

# 回帰器を評価する関数
def eval_on_features(features, target, regressor):
  X_train, X_test = features[:n_train], features[n_train:]
  y_train, y_test = target[:n_train], target[n_train:]
  regressor.fit(X_train, y_train)

  print("Test-set R^2: {}".format(regressor.score(X_test, y_test)))
  y_pred = regressor.predict(X_test)
  y_pred_train = regressor.predict(X_train)
  plt.figure(figsize=(10, 3))
  plt.xticks(range(0, len(X), 8), xticks.strftime("%a %m-%d"), rotation=90, ha="left")

  plt.plot(range(n_train), y_train, label="train")
  plt.plot(range(n_train, len(y_test) + n_train), y_test, '-', label="test")
  plt.plot(range(n_train), y_pred_train, '--', label="predict train")
  plt.plot(range(n_train, len(y_test) + n_train), y_pred, '--', label="predict test")
  plt.legend(loc=(1.01, 0))
  plt.xlabel("Date")
  plt.ylabel("Rentals")


reg = RandomForestRegressor(n_estimators=100, random_state=0)
# posix timeでやっても、決定木の外であること、また、そもそも経過秒を特徴量にしても意味ないので
# eval_on_features(X, y, reg)

# 時刻だけを特徴量としてみる
# X_hours = citibike.index.hour.to_numpy().reshape(-1, 1)
# eval_on_features(X_hours, y, reg)

# 時刻に加えて曜日も特徴量として渡してみる
X_hour_week = np.hstack([
  citibike.index.dayofweek.to_numpy().reshape(-1, 1),
  citibike.index.hour.to_numpy().reshape(-1, 1)
])
eval_on_features(X_hour_week, y, reg)
plt.show()
