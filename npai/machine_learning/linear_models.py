from numpy import ndarray
from .base import *
import numpy as np
from typing import Optional

class LinearRegression(Estimator):
    def __init__(self, closed: bool = True, bias: bool=True, max_iters: Optional[int] = None, lr: Optional[float] = None) -> None:
        super().__init__()

        self.bias = bias
        self.closed = closed
        self.max_iters = max_iters
        self.lr = lr

    def fit(self, X: np.ndarray, y: np.ndarray) -> Estimator:
        N, D = X.shape
        if D > N:
            raise UserWarning("Insufficient number of data points")
        
        if self.bias:
            D += 1
            X = np.hstack((X, np.ones(N).reshape((-1,1))))

        if self.closed:
            if D > N:
                mat = np.linalg.pinv(X @ X.T)
                w = X.T @ mat @ y
            else:
                mat = np.linalg.pinv(X.T @ X)
                w = mat @ X.T @ y
        else:
            w = np.zeros(D)
            for t in range(self.max_iters):
                grad = 2 * X.T @ X @ w - 2 * y @ X
                print(grad)
                w -= self.lr * grad
        
        self.w = w[:-1]
        self.b = w[-1]
        
        return self
    
    def transform(self, X: np.ndarray) -> ndarray:
        y = X @ self.w + self.b
        return y
    
    @property
    def feature_importances_(self) -> np.ndarray:
        importances = self.w / np.sum(self.w)
        return importances