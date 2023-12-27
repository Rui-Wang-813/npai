from .base import *
from numpy import ndarray
import numpy as np
from typing import Optional
from tqdm import tqdm

import npai.optimization as npop


def sigmoid(z):
    return np.clip(1 / (1 + np.exp(-z)), 1e-16, 1 - 1e-16)


class LogisticRegression(Estimator):

    def __init__(self, 
                 maxiter: int = 1000, 
                 eps: float = 0.001, 
                 lr: float = 0.01, 
                 norm: bool=False, 
                 lambda_reg: float = 0.01
                 ) -> None:
        """
        :param bias: whether to use bias in prediction
        """
        super().__init__()

        self.maxiter = maxiter
        self.eps = eps
        self.lr = lr
        self.norm = norm 
        self.lambda_reg = lambda_reg
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool=False) -> Estimator:
        """
        fit the weights and bias according to provided data and targets
        :param X: train data
        :param y: train targets
        """
        self.w, self.b = self.BatchGD(X, y, verbose)

        return self

    def BatchGD(self, X: np.ndarray, y: np.ndarray, verbose: bool=False):
        """
        Batch Gradient Descent for computing weight and bias
        
        :param X: train data
        :param y: train targets
        :param verbose: whether model will print loss every tenth iteration
        """

        w_t = np.zeros(X.shape[1])
        b = 0.1 
        t = 0

        for t in (pbar := tqdm(range(self.maxiter))):

            pred = sigmoid(np.dot(X, w_t) + b)

            #calculate loss
            if self.norm:
                loss = -np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred)) \
                        + self.lambda_reg * np.linalg.norm(w_t)
            else:
                loss = -np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred))
            
            #update w and b
            error = pred - y
            if self.norm:
                w_t -= self.lr * (np.dot(error, X) + 2 * self.lambda_reg * w_t) 
            else:
                w_t -= self.lr * (np.dot(error, X))
            b -= self.lr * np.sum(error)

            #print result
            if verbose:
                if t % 10 == 0:
                    tqdm.write(f"Iteration: {t}, loss: {loss}")

            pbar.set_description(f"Loss: {loss:.5f}")
            if np.linalg.norm(w_t) < self.eps:
                tqdm.write("Converged within eps")
                break
            
        return w_t, b 
    
    def transform(self, X: np.ndarray) -> ndarray:
        """
        transform the prediction data
        :param X: test data to predict
        """
        y = sigmoid(X @ self.w + self.b)
        y[y >= 0.5] = 1
        y[y < 0.5] = 0
        return y
            

        








    
