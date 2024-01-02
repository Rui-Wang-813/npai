from .base import *
from numpy import ndarray
import numpy as np
from typing import Any, Optional

import npai.optimization as npop

import warnings
from tqdm import tqdm

class _Standardizer(Estimator):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X: np.ndarray) -> Estimator:
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self
    
    def transform(self, X: ndarray) -> ndarray:
        return (X - self.mean) / self.std

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
            self.standardizer = _Standardizer().fit(X)
            X = self.standardizer.transform(X)
        
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
            X = self.standardizer.transform(X)

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
        self.standardizer = _Standardizer().fit(X)
        
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
        X = self.standardizer.transform(X)

        y = X @ self.w
        if self.bias:
            y += self.b
        return y
    
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
            self.standardizer = _Standardizer().fit(X)
            X = self.standardizer.transform(X)
        
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
            X = self.standardizer.transform(X)

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
    
"""
Below are implementations of Support Vector Machine (SVM).
Reference: https://www.kaggle.com/code/prabhat12/svm-from-scratch
"""

class SoftSVM(Estimator):
    """
    Soft SVM problem formulated as:
    min lambda_ * |w|^2 + sum_i max(0, 1 - yi(Xi @ w + b))
    Calculate its gradient w.r.t w and perform gradient descent

    Parameters:

    learning_rate : float
        learning rate during gradient descent
    reg_term : float
        regularization term
    max_iters : int
        maximum number of iterations
    eps : float
        epsilon, the threshold for convergence
    """
    def __init__(self, learning_rate: float = .01, reg_term: float = .01, max_iters: int = 1000, eps: float = 0.) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.reg_term = reg_term
        self.max_iters = max_iters
        self.eps = eps

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> Estimator:
        N, D = X.shape

        # force standardization
        # self.standardizer = _Standardizer().fit(X)
        # X = self.standardizer.transform(X)

        # unsqueeze y
        y = y.reshape((-1,1))

        # add bias
        X = np.hstack((X, np.ones((N, 1))))
        D += 1

        # initialize
        w = np.zeros(D)

        prev_w = np.copy(w)

        # perform gradient descent
        for t in (pbar := tqdm(range(self.max_iters))):
            # for i, xi in enumerate(X):
            #     # calculate the gradient
            #     grad = self.reg_term * w
            #     if y[i] * (w @ xi) < 0:
            #         grad += -y[i] * xi
            
            # pre calculations
            mat = y * X
            vec = mat @ w

            # calculate the mask
            mask = vec < 1
            if not np.any(mask):
                tqdm.write("converged within threshold, early stopping...")
                break

            w[:-1] -= self.learning_rate * self.reg_term * w[:-1]    # do not regularize the bias
            w += self.learning_rate * mat[mask].mean(axis=0)

            # calculate loss, early stopping?
            loss = self._loss_fn(vec, w)
            if np.linalg.norm(prev_w - w, ord=2) < self.eps:
                tqdm.write("converged within threshold, early stopping...")
                break
            prev_w = np.copy(w)

            if verbose and t % 100 == 0:
                tqdm.write(f"Iteration {t} | Loss {loss:.5f}")
            pbar.set_description(f"Loss: {loss:.5f}")
                
        self.w = w[:-1]
        self.b = w[-1]

        return self
    
    def transform(self, X: ndarray) -> ndarray:
        y = (X @ self.w + self.b)
        return np.sign(y)
    
    def _loss_fn(self, vec: np.ndarray, w: np.ndarray) -> float:
        return np.where(1 - vec > 0, 0, 1 - vec).mean() + self.reg_term * np.sum(np.square(w))

class HardSVM(Estimator):
    """
    Hard SVM problem formulated as:
    min sum_i max(0, 1 - yi(Xi @ w + b))
    Calculate its gradient w.r.t w and perform gradient descent

    This is a special case of Soft SVM, but optimized in implmenetation

    Parameters:

    learning_rate : float
        learning rate during gradient descent
    reg_term : float
        regularization term
    max_iters : int
        maximum number of iterations
    eps : float
        epsilon, the threshold for convergence
    """
    def __init__(self, learning_rate: float = .01, reg_term: float = 0., max_iters: int = 1000, eps: float = 0.) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.reg_term = reg_term
        self.max_iters = max_iters
        self.eps = eps

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> Estimator:
        N, D = X.shape

        # force standardization
        # self.standardizer = _Standardizer().fit(X)
        # X = self.standardizer.transform(X)

        # unsqueeze y
        y = y.reshape((-1,1))

        # add bias
        X = np.hstack((X, np.ones((N, 1))))
        D += 1

        # initialize
        w = np.zeros(D)

        prev_w = np.copy(w)

        # perform gradient descent
        for t in (pbar := tqdm(range(self.max_iters))):
            # for i, xi in enumerate(X):
            #     # calculate the gradient
            #     grad = self.reg_term * w
            #     if y[i] * (w @ xi) < 0:
            #         grad += -y[i] * xi
            
            # pre calculations
            mat = y * X
            vec = mat @ w

            # calculate the mask
            mask = vec < 1
            if not np.any(mask):
                tqdm.write("converged within threshold, early stopping...")
                break
            w += self.learning_rate * mat[mask].mean(axis=0)

            # calculate loss, early stopping?
            loss = self._loss_fn(vec)
            if np.linalg.norm(prev_w - w, ord=2) < self.eps:
                tqdm.write("converged within threshold, early stopping...")
                break
            prev_w = np.copy(w)

            if verbose and t % 100 == 0:
                tqdm.write(f"Iteration {t} | Loss {loss:.5f}")
            pbar.set_description(f"Loss: {loss:.5f}")
                
        self.w = w[:-1]
        self.b = w[-1]

        return self
    
    def transform(self, X: ndarray) -> ndarray:
        y = (X @ self.w + self.b)
        return np.sign(y)
    
    def _loss_fn(self, vec: np.ndarray) -> float:
        return np.where(1 - vec > 0, 0, 1 - vec).mean()

class PEGASOS(Estimator):
    """
    Soft SVM problem formulated as:
    min lambda_ * |w|^2 + sum_i max(0, 1 - yi(Xi @ w + b))
    Calculate its gradient w.r.t w and perform gradient descent using PEGASOS algorithm

    This is a special case of Soft SVM, but optimized in implmenetation

    Parameters:

    reg_term : float
        regularization term
    max_iters : int
        maximum number of iterations
    eps : float
        epsilon, the threshold for convergence
    """
    def __init__(self, reg_term: float = .01, max_iters: int = 1000, eps: float = 0.) -> None:
        super().__init__()
        self.reg_term = reg_term
        self.max_iters = max_iters
        self.eps = eps

    def fit(self, X: np.ndarray, y: np.ndarray) -> Estimator:
        N, D = X.shape

        # force standardization
        # self.standardizer = _Standardizer().fit(X)
        # X = self.standardizer.transform(X)

        # unsqueeze y
        y = y.reshape((-1,1))

        # add bias
        X = np.hstack((X, np.ones((N, 1))))
        D += 1

        # initialize
        w = np.zeros(D)

        prev_w = np.copy(w)

        # pre calculation
        a = 1. / np.sqrt(self.reg_term)

        # perform gradient descent
        for t in (pbar := tqdm(range(self.max_iters))):
            # for i, xi in enumerate(X):
            #     # calculate the gradient
            #     grad = self.reg_term * w
            #     if y[i] * (w @ xi) < 0:
            #         grad += -y[i] * xi
            
            eta_t = 1 / self.reg_term / (t + 1)

            # randomly choose a sample
            idx = np.random.choice(range(N))
            Xi = X[idx]
            yi = y[idx]

            # update w
            w[:-1] -= eta_t * self.reg_term * w[:-1]
            if yi * (Xi @ w) < 1:
                # incorrect
                w += eta_t * yi * Xi
            # projection step
            w *= min(1, a / np.linalg.norm(w, ord=2))
            
            if np.linalg.norm(prev_w - w, ord=2) < self.eps:
                tqdm.write("converged within threshold, early stopping...")
                break
            prev_w = np.copy(w)
                
        self.w = w[:-1]
        self.b = w[-1]

        return self
    
    def transform(self, X: ndarray) -> ndarray:
        y = (X @ self.w + self.b)
        return np.sign(y)
    
    def _loss_fn(self, vec: np.ndarray, w: np.ndarray) -> float:
        return np.where(1 - vec > 0, 0, 1 - vec).mean()

class Kernel(object):

    def __init__(self) -> None:
        pass

    def __call__(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        pass

class RBFKernel(Kernel):

    def __init__(self, gamma: float = .1) -> None:
        super().__init__()
        self.gamma = gamma
    
    def __call__(self, X: ndarray, Z: ndarray) -> np.ndarray:
        # make sure (X - Z)_{i, j} = Xi - Zj
        X = X[:, np.newaxis]
        Z = Z[np.newaxis, :]
        return np.exp(-self.gamma * np.linalg.norm(X - Z, ord=2, axis=2) ** 2)

class PolyKernel(Kernel):
    def __init__(self, degree: int = 1) -> None:
        super().__init__()
        self.degree = degree
    
    def __call__(self, X: ndarray, Z: ndarray) -> np.ndarray:
        return np.power(1 + X @ Z.T, self.degree)
    
def _is_Q_PD(Q: np.ndarray) -> bool:
    """
    Check if a matrix Q is positive definite?
    """
    if not np.all(Q == Q.T):
            return False        
    eigv, _ = np.linalg.eig(Q)
    if not np.all(eigv > 0):
        return False
    return True

class DotProdKernel(Kernel):
    def __init__(self, Q: Optional[np.ndarray] = None) -> None:
        super().__init__()
        self.Q = Q
        if Q is not None and not _is_Q_PD(Q):
            raise ValueError("Given matrix must be positive definite!")
    
    def __call__(self, X: ndarray, Z: ndarray) -> ndarray:
        if self.Q is None:
            # no Q given, just use identity
            self.Q = np.eye(X.shape[1])

        return X @ self.Q @ Z.T

class DualSVM(Estimator):
    """
    Soft SVM problem solved using Duality
    max sum_n alpha_n  - 0.5 * sum_{m,n} ym * yn * alpha_m * alpha_n * xm @ xn
    Calculate its gradient w.r.t alpha and perform gradient descent

    This is a special case of Soft SVM, but optimized in implmenetation

    Parameters:

    learning_rate : float
        learning rate
    C : float
        the term used to control slackness of soft SVM
    max_iters : int
        maximum number of iterations
    eps : float
        epsilon, the threshold for convergence
    kernel : Kernel
        the kernel function
    """
    def __init__(self, learning_rate: float = .001, C: float = 1., max_iters: int = 1000, eps: float = 0., opt: str = "BGD", kernel: Kernel = DotProdKernel()) -> None:
        if opt not in ["BGD", "SMO"]:
            raise NotImplementedError("Only support Gradient Descent or SMO optimization")
        
        super().__init__()
        self.learning_rate = learning_rate
        self.C = C
        self.max_iters = max_iters
        self.eps = eps
        self.opt = opt
        self.kernel = kernel

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> Estimator:
        N, D = X.shape

        # force standardization
        # self.standardizer = _Standardizer().fit(X)
        # X = self.standardizer.transform(X)
        self.X = X
        self.y = y

        if self.opt == "BGD":
            self._BGA(verbose)
        else:
            self._SMO(verbose)

        return self
    
    def _H(self, X: np.ndarray) -> np.ndarray:
        return (self.alpha * self.y) @ self.kernel(self.X, X) + self.b

    def transform(self, X: np.ndarray) -> None:
        return np.sign(self._H(X))    

    def _BGA(self, verbose: bool = False) -> np.ndarray:
        # pre calculate yi * yj * K(xi, xj)
        kernel_mat = np.outer(self.y, self.y) * self.kernel(self.X, self.X)

        self.alpha = np.zeros(self.X.shape[0])
        prev_alpha = np.copy(self.alpha)
        for t in (pbar := tqdm(range(self.max_iters))):
            grad = 1 - (kernel_mat @ self.alpha)
            self.alpha += self.learning_rate * grad         # gradient ascent as we are maximizing
            self.alpha = np.clip(self.alpha, 0, self.C)     # KKT: 0 <= alpha <= C
            # alpha -= np.mean(alpha * self.y)    # KKT: sum_n alpha_n * yn = 0

            if np.linalg.norm(prev_alpha - self.alpha, ord=2) < self.eps:
                tqdm.write("converged within threshold, early stopping...")
                break
            prev_alpha = np.copy(self.alpha)

            # calculate loss
            gain = np.sum(self.alpha) - 0.5 * np.sum(np.outer(self.alpha, self.alpha) * kernel_mat)
            if verbose and t % 100 == 0:
                tqdm.write(f"Iteration {t} | Gain {gain:.5f}")
            pbar.set_description(f"Gain: {gain:.5f}")
        
        # recover b
        mask = (self.alpha > 0) & (self.alpha < self.C)
        yn = self.y[mask]
        Xn = self.X[mask]
        self.b = (yn - (self.alpha *self.y) @ self.kernel(self.X, Xn)).mean()
    
    def _SMO(self, verbose: bool = False) -> None:
        self.alpha = np.zeros(self.X.shape[0])
        self.b = 0

        XX = self.X @ self.X.T

        t = 0
        while t < self.max_iters:
            num_changed_alphas = 0
            for i in range(self.X.shape[0]):
                # calculate difference between prediction and true label
                Ei = self._H(self.X[i]) - self.y[i]
                if (self.y[i] * Ei < -self.eps and self.alpha[i] < self.C) or \
                    (self.y[i] * Ei > self.eps and self.alpha[i] > 0):
                    # randomly choose j != i
                    j = np.random.choice(range(self.X.shape[0] - 1))
                    if j >= i: j += 1

                    # calculate Ej
                    Ej = self._H(self.X[j]) - self.y[j]

                    # save old alpha's
                    alphas_old = (self.alpha[i], self.alpha[j])

                    # compute lower bound
                    if self.y[i] != self.y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    
                    if L == H:
                        continue

                    # compute eta
                    eta = 2 * XX[i,j] - XX[i,i] - XX[j,j]

                    # update alpha_j
                    self.alpha[j] -= self.y[j] * (Ei - Ej) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    if np.abs(self.alpha[j] - alphas_old[1]) < 1e-5:
                        continue

                    # update alpha_i
                    self.alpha[i] += self.y[i] * self.y[j] * (alphas_old[1] - self.alpha[j])

                    # calculate b1 and b2
                    b1 = self.b - Ei - self.y[i] * (self.alpha[i] - alphas_old[0]) * XX[i,i] - self.y[j] * (self.alpha[j] - alphas_old[1]) * XX[i,j]
                    b2 = self.b - Ej - self.y[i] * (self.alpha[i] - alphas_old[0]) * XX[i,j] - self.y[j] * (self.alpha[j] - alphas_old[1]) * XX[j,j]
                    if self.alpha[i] >= 0 and self.alpha[i] <= self.C:
                        self.b = b1
                    elif self.alpha[j] >= 0 and self.alpha[j] <= self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.

                    # update num changed alphas
                    num_changed_alphas += 1
                
            if num_changed_alphas:
                t = 0
            else:
                t += 1