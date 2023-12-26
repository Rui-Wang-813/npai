from .base import *
from numpy import ndarray
import numpy as np
from typing import Optional

import npai.optimization as npop

class LinearRegression(Estimator):
    def __init__(self, bias: bool=True) -> None:
        """
        :param bias: whether to use bias in prediction
        """
        super().__init__()

        self.bias = bias

    def fit(self, X: np.ndarray, y: np.ndarray) -> Estimator:
        """
        fit the weights and bias according to provided data and targets
        :param X: train data
        :param y: train targets
        """
        N, D = X.shape
        if D > N:
            # more features than number of samples?
            raise UserWarning("Insufficient number of data points")
        
        if self.bias:
            # append bias to the weight calculation
            D += 1
            X = np.hstack((X, np.ones(N).reshape((-1,1))))

        if D > N:
            # use X^T(XX^T)^(-1)y
            mat = np.linalg.pinv(X @ X.T)
            w = X.T @ mat @ y
        else:
            # use (X^TX)^(-1)X^Ty
            mat = np.linalg.pinv(X.T @ X)
            w = mat @ X.T @ y
        
        self.w = w[:-1]
        self.b = w[-1]
        
        return self
    
    def transform(self, X: np.ndarray) -> ndarray:
        """
        transform the prediction data
        :param X: test data to predict
        """
        y = X @ self.w + self.b
        return y
    
    @property
    def coef_(self) -> np.ndarray:
        return self.w
    
    @property
    def bias_(self) -> float:
        return self.b
    
    @property
    def feature_importances_(self) -> np.ndarray:
        importances = self.w / np.sum(self.w)
        return importances