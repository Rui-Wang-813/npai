import sys
sys.path.append("../")
sys.path.append("../npai/")

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles

X1, y1 = make_blobs(n_samples=200, centers=2, random_state=0, cluster_std=0.60)
y1 = np.where(y1 <= 0, -1, 1)
plt.scatter(X1[:, 0], X1[:, 1], c=y1, s=50, cmap='winter', alpha=.5)
plt.title("Dataset 1")
plt.show()

import npai.machine_learning as npml

svm = npml.PEGASOS(max_iters=5000, eps=1e-15)
svm.fit(X1, y1)
preds = svm.transform(X1)
acc = (preds == y1).mean()
print(f"prediction accuracy is: {acc:.5f}")

w = svm.w
b = svm.b

slope = -w[0] / w[1]
y_intercept = -b / w[1]

x_space = np.linspace(np.min(X1[:, 0]), np.max(X1[:, 0]), num=1000)
ys = slope * x_space + y_intercept
plt.plot(x_space, ys)
plt.scatter(X1[:, 0], X1[:, 1], c=y1, s=50, cmap='winter', alpha=.5)
plt.show()