import sys
sys.path.append("..")
sys.path.append("../npai/")

from npai.machine_learning import LinearRegression, RidgeRegression, LassoRegression
import numpy as np
import matplotlib.pyplot as plt

X = np.arange(100).reshape((-1, 1))
y = 0.1 * X.reshape((-1,)) + 5 + np.random.randn(100)
# X = (X - np.mean(X, keepdims=True)) / np.std(X, keepdims=True)
# X = np.random.randn(3, 10)
# y = X @ np.array([i for i in range(10)])

model = LassoRegression(bias=True, max_iters=2000, learning_rate=0.01, eps=1e-15)
model.fit(X, y, verbose=True)

preds = model.transform(X)
print(np.mean((preds - y) ** 2))

print(model.feature_importances_)
print(model.coef_)
print(model.bias_)

plt.scatter(X, y, c="blue")
plt.plot(X, preds)
plt.show()