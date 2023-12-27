from .base import *
from numpy import ndarray
import numpy as np
from typing import Optional

import npai.optimization as npop

import warnings
from tqdm import tqdm

class LinearRegression(Estimator):
    def __init__(self, bias: bool=True, closed: bool = True, max_iters: Optional[int] = None, learning_rate: Optional[float] = None,
                 eps: Optional[float] = 1e-15, regularization: Optional[str] = None, reg_term: Optional[float] = None) -> None:
        """
        Parameters:

        bias : bool
            whether to use bias
        closed : bool
            whether to use closed form solution
        max_iters : Optional[int]
            if not closed, then max_iters is the maximum number of iterations
        learning_rate : Optional[float]
            if not closed, then this is the learning rate
        eps: Optional[float]
            if not closed, then eps is the stopping criteria
        regularization : Optional[str]
            the regularization applied on weights, can be "ridge", "lasso" or None
        reg_term : Optional[float]
            the regularization term
        """
        super().__init__()

        self.bias = bias
        self.closed = closed
        self.max_iters = max_iters
        self.learning_rate = learning_rate
        self.eps = eps
        self.regularization = regularization
        self.reg_term = reg_term

        if regularization not in ["lasso", "ridge", None]:
            raise NotImplementedError("Only support LASSO or Ridge regression")
        if self.closed and self.regularization == "lasso":
            raise ValueError("LASSO regression does not have closed-form solution")

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> Estimator:
        """
        fit the weights and bias according to provided data and targets

        Parameters:
        X : np.ndarray
            train data
        y : np.ndarray
            train targets
        verbose : bool 
            whether to print training info for GD version
        """
        N, D = X.shape
        if D > N:
            # more features than number of samples?
            warnings.warn("Insufficient number of data points")
        
        if not self.closed:
            # force to normalize it for GD version
            X = self._data_normalize(X)
        
        if self.bias:
            # append bias to the weight calculation
            D += 1
            X = np.hstack((X, np.ones(N).reshape((-1,1))))

        if self.closed:
            # use closed form solution
            if D > N:
                # use X^T(XX^T + lamb * I)^(-1)y
                if self.regularization == "ridge":
                    I = np.eye(N)
                    I[-1, -1] = 0
                    mat = np.linalg.pinv(X @ X.T + self.reg_term * I)
                else:
                    mat = np.linalg.pinv(X @ X.T)
                w = X.T @ mat @ y
            else:
                # use (X^TX + lamb * I)^(-1)X^Ty
                if self.regularization == "ridge":
                    I = np.eye(D)
                    I[-1, -1] = 0
                    mat = np.linalg.pinv(X.T @ X + self.reg_term * I)
                else:
                    mat = np.linalg.pinv(X.T @ X)
                w = mat @ X.T @ y
        else:
            # use batch gradient descent
            w = self._BGD(X, y, verbose)
        
        if self.bias:
            self.w = w[:-1]
            self.b = w[-1]
        else:
            self.w = w
        
        return self
    
    def transform(self, X: np.ndarray) -> ndarray:
        """
        transform the prediction data
        :param X: test data to predict
        """
        if not self.closed:
            X = self._data_normalize(X, train=False)

        y = X @ self.w
        if self.bias:
            y += self.b
        return y
    
    def _BGD(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> np.ndarray:
        N, D = X.shape
        w = np.zeros(D)
        for t in (pbar := tqdm(range(self.max_iters))):
            # perform prediction
            preds = X @ w
            # calculate loss
            loss = np.mean((preds - y) ** 2)
            if loss < self.eps:
                tqdm.write("converged within threshold, early stopping")
                break

            grad = 2 * (X.T @ (preds - y)) / N
            # perform regularization
            if self.regularization == "ridge":
                grad[:-1] += 2 * self.reg_term * w[:-1]
            elif self.regularization == "lasso":
                grad[:-1] += self.reg_term * np.sign(w[:-1])
            w -= self.learning_rate * grad

            if verbose:
                if t % 100 == 0:
                    tqdm.write(f"Iteration: {t}, loss: {loss}")
            pbar.set_description(f"Loss: {loss:.5f}")
        
        return w
    
    def _data_normalize(self, X: np.ndarray, train: bool = True) -> np.ndarray:
        """
        Perform normalization on the given data

        Parameters:

        X : np.ndarray
            the data to normalize
        train : bool
            whether this is training time. if true, then store the mean and std of data
        """
        if train:
            # only calculate and use mean and std of X if is training
            self.data_mean = np.mean(X, axis=0)
            self.data_std = np.std(X, axis=0)

        return (X - self.data_mean) / self.data_std
    
    @property
    def coef_(self) -> np.ndarray:
        return self.w
    
    @property
    def bias_(self) -> float:
        if not self.bias:
            raise AttributeError("Unbiased linear regression does not have bias")
        return self.b
    
    @property
    def feature_importances_(self) -> np.ndarray:
        weights = np.abs(self.w)
        importances = weights / np.sum(weights)
        return importances