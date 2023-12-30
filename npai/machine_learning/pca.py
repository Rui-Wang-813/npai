from .base import *
import numpy as np


class PCA(Estimator):
    """
    Principal Component Analysis (PCA)
    """
    def __init__(self, n_components=None):
        """
        :param n_components: int, the number of components to keep
        """
        self.n_components = n_components
        self.components = None
        self.explained_variance = None

    def fit(self, X):
        """
        Fit the model with X.
        :param X: array-like of shape (n_samples, n_features)
        :param y: array-like of shape (n_samples,) or (n_samples, n_targets)
        :param verbose: bool, whether to print the loss
        :return: self
        """
        X = X - np.mean(X, axis=0)

        covMat = np.divide(np.matmul(X.T, X), X.shape[0] )
        eigenvalues, eigenvectors = np.linalg.eig(covMat)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]
        self.components = eigenvectors[:, :self.n_components]

        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        X_pca = X @ self.components
        return X_pca
    