from .base import *
from numpy import ndarray
import numpy as np
from typing import Optional
from tqdm import tqdm

import npai.optimization as npop


def sigmoid(z):
    return np.clip(1 / (1 + np.exp(-z)), 1e-16, 1 - 1e-16)

def softmax(z):
    _exps = np.exp(z)
    _probs = _exps / np.sum(_exps, axis=1, keepdims=True)
    return np.clip(_probs, 1e-16, 1 - 1e-16)

def one_hot(y):
    n_classes = np.unique(y).shape[0]
    y_encoded = np.zeros((y.shape[0], n_classes))
    np.put_along_axis(y_encoded, y.reshape((-1,1)), 1, axis=1)
    return y_encoded


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
            

        

class MulticlassLogisticRegression(Estimator):

    def __init__(self, 
                 maxiter: int = 10_000, 
                 eps: float = 0.001, 
                 lr: float = 0.1, 
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
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        y_encoded = one_hot(y)
        w_t = np.zeros((y_encoded.shape[1], X.shape[1]))
        t = 0

        prev_loss = float("inf")
        for t in (pbar := tqdm(range(self.maxiter))):

            outputs = softmax(X @ w_t.T)

            #calculate loss
            loss = -np.sum(np.log(outputs) * y_encoded)
            if self.norm:
                # helper variable since I am not regularizing the bias term
                _w_t = w_t[:, :-1]
                loss += self.lambda_reg * np.sum(_w_t ** 2)
            
            #update w and b
            error = outputs - y_encoded
            grad = error.T @ X / X.shape[0]
            if self.norm:
                grad[:, :-1] += 2 * _w_t

            w_t -= self.lr * grad

            #print result
            if verbose:
                if t % 10 == 0:
                    tqdm.write(f"Iteration: {t}, loss: {loss}")

            pbar.set_description(f"Loss: {loss:.5f}")
            # if np.linalg.norm(w_t) < self.eps:
            #     tqdm.write("Converged within eps")
            #     break
            if np.abs(prev_loss - loss) < self.eps:
                tqdm.write("Converged within eps")
                break
            
        return w_t[:, :-1], w_t[:, -1]
    
    def transform(self, X: np.ndarray) -> ndarray:
        """
        transform the prediction data
        :param X: test data to predict
        """
        outputs = softmax(X @ self.w.T + self.b)
        predictions = np.argmax(outputs, axis=1)
        return predictions