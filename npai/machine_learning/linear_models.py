from .base import *
from numpy import ndarray
import numpy as np
from typing import Optional

import npai.optimization as npop

import warnings
from tqdm import tqdm

class LinearRegression(Estimator):
    def __init__(self, bias: bool=True, closed: bool = True, max_iters: Optional[int] = None, learning_rate: Optional[float] = None,
                 eps: Optional[float] = None) -> None:
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
        eps : Optional[float]
            if not closed, then eps is the stopping criteria
        """
        super().__init__()

        self.bias = bias
        self.closed = closed
        self.max_iters = max_iters
        self.learning_rate = learning_rate
        self.eps = eps

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
            X = np.hstack((X, np.ones(N).reshape((-1,1))))

        if self.closed:
            # use closed form solution
            if D > N:
                # use X^T(XX^T)^(-1)y
                mat = np.linalg.pinv(X @ X.T)
                w = X.T @ mat @ y
            else:
                # use (X^TX)^(-1)X^Ty                
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

        prev_loss = float("inf")
        for t in (pbar := tqdm(range(self.max_iters))):
            # perform prediction
            preds = X @ w
            # calculate loss
            loss = np.mean((preds - y) ** 2)
            if np.abs(prev_loss - loss) < self.eps:
                tqdm.write("converged within threshold, early stopping")
                break

            grad = 2 * (X.T @ (preds - y)) / N
            w -= self.learning_rate * grad

            if verbose:
                if t % 100 == 0:
                    tqdm.write(f"Iteration: {t}, loss: {loss}")
            pbar.set_description(f"Loss: {loss:.5f}")

            prev_loss = float("inf")
        
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

class LassoRegression(Estimator):
    def __init__(self, bias: bool=True, max_iters: int = 100, learning_rate: float = .01,eps: float = 1e-15, reg_term: float = .01) -> None:
        """
        Parameters:

        bias : bool
            whether to use bias
        max_iters : Optional[int]
            if not closed, then max_iters is the maximum number of iterations
        learning_rate : float
            learning rate
        eps : Optional[float]
            stopping criteria
        reg_term : Optional[float]
            the regularization term
        """
        super().__init__()

        self.bias = bias
        self.max_iters = max_iters
        self.learning_rate = learning_rate
        self.eps = eps
        self.reg_term = reg_term

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
        
        # force normalization as we are using GD
        X = self._data_normalize(X)
        
        if self.bias:
            # append bias to the weight calculation
            X = np.hstack((X, np.ones(N).reshape((-1,1))))

        # use batch gradient descent
        w = np.zeros(X.shape[1])
        prev_loss = float("inf")
        for t in (pbar := tqdm(range(self.max_iters))):
            # perform prediction
            preds = X @ w
            # calculate loss
            loss = np.mean((preds - y) ** 2)
            if np.abs(prev_loss - loss) < self.eps:
                tqdm.write("converged within threshold, early stopping")
                break

            grad = 2 * (X.T @ (preds - y)) / N + self.reg_term * np.sign(w[:D])
            w -= self.learning_rate * grad

            if verbose:
                if t % 100 == 0:
                    tqdm.write(f"Iteration: {t}, loss: {loss}")
            pbar.set_description(f"Loss: {loss:.5f}")     
            
            prev_loss = loss       
        
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
        # force normalization
        X = self._data_normalize(X, train=False)

        y = X @ self.w
        if self.bias:
            y += self.b
        return y
    
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

class RidgeRegression(Estimator):
    def __init__(self, bias: bool=True, closed: bool = True, max_iters: Optional[int] = None, learning_rate: Optional[float] = None,
                 eps: Optional[float] = None, reg_term: float = .01) -> None:
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
        eps : Optional[float]
            if not closed, then eps is the stopping criteria
        reg_term : float
            regularization term
        """
        super().__init__()

        self.bias = bias
        self.closed = closed
        self.max_iters = max_iters
        self.learning_rate = learning_rate
        self.eps = eps
        self.reg_term = reg_term

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
            # use (X^TX + lambd * I)^(-1)X^Ty
            I = np.eye(D)
            if self.bias:
                I[-1, -1] = 0.      
            mat = np.linalg.inv(X.T @ X + self.reg_term * I)
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
            # force normalization for GD version
            X = self._data_normalize(X, train=False)

        y = X @ self.w
        if self.bias:
            y += self.b
        return y
    
    def _BGD(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> np.ndarray:
        N, D = X.shape
        w = np.zeros(D)

        prev_loss = float("inf")
        for t in (pbar := tqdm(range(self.max_iters))):
            # perform prediction
            preds = X @ w
            # calculate loss
            loss = np.mean((preds - y) ** 2)
            if np.abs(prev_loss - loss) < self.eps:
                tqdm.write("converged within threshold, early stopping")
                break

            grad = 2 * ((X.T @ (preds - y)) / N + self.reg_term * w)
            w -= self.learning_rate * grad

            if verbose:
                if t % 100 == 0:
                    tqdm.write(f"Iteration: {t}, loss: {loss}")
            pbar.set_description(f"Loss: {loss:.5f}")

            prev_loss = loss
        
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