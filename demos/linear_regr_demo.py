import sys
sys.path.append("..")
sys.path.append("../npai/")

from npai.machine_learning import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

X = np.arange(100).reshape((-1, 1))
y = 0.1 * X.reshape((-1,)) + 5 + np.random.randn(100)

model = LinearRegression()
model.fit(X, y)

preds = model.transform(X)
print(np.sum((preds - y) ** 2))
print(model.w, model.b)

plt.scatter(X, y, c="blue")
plt.plot(X, preds)
plt.show()