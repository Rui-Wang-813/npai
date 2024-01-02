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

# X1, y1 = make_moons(n_samples=200, noise=.05)
# y1 = np.where(y1 <= 0, -1, 1)
# print("First five rows and col values \nX2 : \n",X1[:5], " \n y2 :\n",y1[:5])
# plt.scatter(X1[:, 0], X1[:, 1], c=y1, s=50, cmap='winter', alpha=.5)
# plt.title("Dataset 2")
# plt.show()

# X1, y1 = noisy_circles = make_circles(n_samples=200, factor=.5, noise=.05)
# y1 = np.where(y1 <= 0, -1, 1)
# print("First five rows and col values \nX1 : \n",X1[:5], " \n y3 :\n",y1[:5])
# plt.scatter(X1[:, 0], X1[:, 1], c=y1, s=50, cmap='winter', alpha=.5)
# plt.title("Dataset 3")
# plt.show()

import npai.machine_learning as npml

svm = npml.DualSVM(max_iters=1000, eps=1e-15, learning_rate=.001, C=1., opt="SMO", kernel=npml.RBFKernel())
svm.fit(X1, y1)
preds = svm.transform(X1)
acc = (preds == y1).mean()
print(f"prediction accuracy is: {acc:.5f}")

plt.scatter(X1[:, 0], X1[:, 1], c=y1, s=50, cmap='winter', alpha=.5)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 50)
yy = np.linspace(ylim[0], ylim[1], 50)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = ((svm.alpha * svm.y) @ svm.kernel(svm.X, xy) + svm.b).reshape(XX.shape)
ax.contour(XX, YY, Z, levels=[-1, 0, 1],linestyles=['--', '-', '--'])
plt.show()