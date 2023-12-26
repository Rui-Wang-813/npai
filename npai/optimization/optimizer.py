"""18-661 HW5 Optimization Policies."""

import numpy as np

from .base import Optimizer


class SGD(Optimizer):
    """Simple SGD optimizer.

    Parameters
    ----------
    learning_rate : float
        SGD learning rate.
    """

    def __init__(self, learning_rate=0.01):

        self.learning_rate = learning_rate

    def apply_gradients(self, params):
        """Apply gradients to parameters.

        Parameters
        ----------
        params : Variable[]
            List of parameters that the gradients correspond to.
        """
        for param in params:
            param.value -= self.learning_rate * param.grad

class ASGD(Optimizer):
    """Averaged SGD optimizer.

    Parameters
    ----------
    learning_rate : float
        SGD learning rate.
    t0: int
        the starting point to do averaging
    """

    def __init__(self, learning_rate=0.01, t0: int = 0, T: int = 0):

        self.learning_rate = learning_rate
        self.t0 = t0
        self.T = T
    
    def initialize(self, params):
        self.avgs = [np.zeros_like(param) for param in params]
        self.t = 0

    def apply_gradients(self, params):
        """Apply gradients to parameters.

        Parameters
        ----------
        params : Variable[]
            List of parameters that the gradients correspond to.
        """
        for i, param in enumerate(params):
            param.value -= self.learning_rate * param.grad

            if self.t >= self.t0:
                self.avgs[i] = (self.avgs[i] * (self.t - self.t0) + param.value) / (self.t - self.t0 + 1)
            if self.t == self.T - 1:
                param.value = self.avgs[i]

        self.t += 1

class Adam(Optimizer):
    """Adam (Adaptive Moment) optimizer.

    Parameters
    ----------
    learning_rate : float
        Learning rate multiplier.
    beta1 : float
        Momentum decay parameter.
    beta2 : float
        Variance decay parameter.
    epsilon : float
        A small constant added to the demoniator for numerical stability.
    weight_decay : float
        A small constant used to decay the weight of parameter
    amsgrad : bool
        Whether to use AMSGrad variant of the optimizer
    """

    def __init__(
            self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7, weight_decay: float = 0., amsgrad: bool = False):

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

    def initialize(self, params):
        """Initialize any optimizer state needed.

        params : np.array[]
            List of parameters that will be used with this optimizer.
        """
        self.m = [np.zeros_like(param.value) for param in params]
        self.v = [np.zeros_like(param.value) for param in params]
        if self.amsgrad:
            self.vmax = [np.zeros_like(param.value) for param in params]

    def apply_gradients(self, params):
        """Apply gradients to parameters.

        Parameters
        ----------
        params : Variable[]
            List of parameters that the gradients correspond to.
        """
        for i, param in enumerate(params):
            grad = param.grad + self.weight_decay * param.value
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2
            m_hat = self.m[i] / (1 - self.beta1)
            v_hat = self.v[i] / (1 - self.beta2)
            if self.amsgrad:
                self.vmax[i] = np.where(self.vmax[i] > v_hat, self.vmax[i], v_hat)
                v_hat = self.vmax[i]
            param.value -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

class Adagrad(Optimizer):

    def __init__(self, learning_rate=0.01, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.epsilon = epsilon

    def initialize(self, params):
        self.G = [np.zeros_like(param.value) for param in params]

    def apply_gradients(self, params):
        for i, param in enumerate(params):
            self.G[i] += param.grad**2
            param.value -= self.learning_rate * param.grad / (np.sqrt(self.G[i]) + self.epsilon)


class Adadelta(Optimizer):

    def __init__(self, learning_rate=1.0, rho=0.95, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon

    def initialize(self, params):
        self.G = [np.zeros_like(param.value) for param in params]
        self.delta = [np.zeros_like(param.value) for param in params]

    def apply_gradients(self, params):
        for i, param in enumerate(params):
            self.G[i] = self.rho * self.G[i] + (1 - self.rho) * param.grad**2
            param.value -= self.learning_rate * param.grad / (np.sqrt(self.G[i]) + self.epsilon)


class Adamax(Optimizer):

    def __init__(self, learning_rate=1.0, beta1: float = 0.9, beta2: float = 0.999, epsilon=1e-7, weight_decay: float = 0.):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

    def initialize(self, params):
        self.m = [np.zeros_like(param.value) for param in params]
        self.u = [np.zeros_like(param.value) for param in params]
        self.t = 1

    def apply_gradients(self, params):
        for i, param in enumerate(params):
            grad = param.grad + self.weight_decay * param.value
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad            
            a1 = self.beta2 * self.u[i]
            a2 = np.abs(grad) + self.epsilon
            self.u[i] = np.where(a1 > a2, a1, a2)
            param.value -= self.learning_rate * self.m[i] / ((1 - self.beta1 ** self.t) * self.u[i])
        
        self.t += 1

class AdamW(Optimizer):
    """AdamW (Adaptive Moment) optimizer with fixed weight decay.

    Parameters
    ----------
    learning_rate : float
        Learning rate multiplier.
    beta1 : float
        Momentum decay parameter.
    beta2 : float
        Variance decay parameter.
    epsilon : float
        A small constant added to the demoniator for numerical stability.
    weight_decay : float
        A small constant used to decay the weight of parameter
    amsgrad : bool
        Whether to use AMSGrad variant of the optimizer
    """

    def __init__(
            self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7, weight_decay: float = 0.01, amsgrad: bool = False):

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

    def initialize(self, params):
        """Initialize any optimizer state needed.

        params : np.array[]
            List of parameters that will be used with this optimizer.
        """
        self.m = [np.zeros_like(param.value) for param in params]
        self.v = [np.zeros_like(param.value) for param in params]
        if self.amsgrad:
            self.vmax = [np.zeros_like(param.value) for param in params]

    def apply_gradients(self, params):
        """Apply gradients to parameters.

        Parameters
        ----------
        params : Variable[]
            List of parameters that the gradients correspond to.
        """
        for i, param in enumerate(params):
            param.value *= (1. - self.learning_rate * self.weight_decay)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * param.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * param.grad ** 2
            m_hat = self.m[i] / (1 - self.beta1)
            v_hat = self.v[i] / (1 - self.beta2)
            if self.amsgrad:
                self.vmax[i] = np.where(self.vmax[i] > v_hat, self.vmax[i], v_hat)
                v_hat = self.vmax[i]
            param.value -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)