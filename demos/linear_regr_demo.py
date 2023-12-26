import sys
sys.path.append("..")
sys.path.append("../npai/")

from npai.machine_learning.linear_models import LinearRegression
import numpy as np

X = np.arange(100).reshape((-1, 1))
y = 2 * X.reshape((-1,)) + 5

model = LinearRegression(closed=False, max_iters=100, lr=1e-3)
model.fit(X, y)

preds = model.transform(X)
print(np.sum((preds - y) ** 2))
print(model.w, model.b)