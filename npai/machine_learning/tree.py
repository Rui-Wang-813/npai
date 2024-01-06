from .base import *
import numpy as np

def _gini_impurity(y: np.ndarray) -> float:
    y_counts = np.unique(y, return_counts=True)[1]
    y_probs = y_counts / y.shape[0]
    return np.sum(y_probs * (1. - y_probs))

def _entropy(y: np.ndarray) -> float:
    y_counts = np.unique(y, return_counts=True)[1]
    y_probs = y_counts / y.shape[0]
    return -np.sum(y_probs * np.log2(y_probs))

def _mean_squared_err(y: np.ndarray) -> float:
    y_mean = np.mean(y)
    return np.mean((y - y_mean) ** 2)

def _mean_absolute_err(y: np.ndarray) -> float:
    y_median = np.median(y)
    return np.mean(np.abs(y - y_median))

def _poisson_deviance(y: np.ndarray) -> float:
    y_mean = np.mean(y)
    return np.mean(y * np.log(y / y_mean) - y + y_mean)

class _DecisionNode(Estimator):

    def __init__(self, max_depth: int = 1, split_rule: str = "gini", is_clf: bool = True) -> None:
        super().__init__()
        self.max_depth = max_depth
        self.is_clf = is_clf
        if split_rule == "gini":
            self.split_measure = _gini_impurity
        elif split_rule == "entropy":
            self.split_measure = _entropy
        elif split_rule == "mse":
            self.split_measure = _mean_squared_err
        elif split_rule == "mae":
            self.split_measure = _mean_absolute_err
        else:
            self.split_measure = _poisson_deviance

        self.split_rule = split_rule
        self.is_leaf = False

    def _get_pred(self, y: np.ndarray) -> None:
        self.is_leaf = True
        if self.is_clf:
            # perform classification
            ys, cnts = np.unique(y, return_counts=True)
            self.pred = ys[np.argmax(cnts)]
        else:
            # perform regression
            self.pred = np.mean(y)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Estimator:
        if self.max_depth == 0 or np.unique(y).shape[0] <= 1:
            # leaf node
            self._get_pred(y)
            return self

        axs = np.arange(X.shape[1])
        measure_val = self.split_measure(y)

        # find the best feature, threshold according to gains
        best_stats = (-1, np.nan, float("-inf"))    # axis, threshold, gain
        for ax in axs:
            ax_vals = X[:, ax]
            # construct q + 1 thresholds
            thresholds = np.sort(np.unique(ax_vals))
            thresholds = np.append(thresholds, thresholds[-1] + 1)
            thresholds = np.append(thresholds[0] - 1, thresholds)
            thresholds = (thresholds[:-1] + thresholds[1:]) / 2.

            for pivot in thresholds:
                mask = ax_vals < pivot

                ltys = y[mask]
                rtys = y[~mask]

                # calculate gains
                measure1 = self.split_measure(ltys)
                measure2 = self.split_measure(rtys)
                gain = measure_val - (ltys.shape[0] * measure1 + rtys.shape[0] * measure2) / y.shape[0]
                if gain > best_stats[2]:
                    best_stats = (ax, pivot, gain)
        
        self.ax, self.threshold, _ = best_stats
        mask = X[:, self.ax] < self.threshold
        if np.all(mask) or not np.any(mask):
            # meaningless split
            self._get_pred(y)
            return self

        self.left = _DecisionNode(self.max_depth - 1, self.split_rule).fit(X[mask], y[mask])
        self.right = _DecisionNode(self.max_depth - 1, self.split_rule).fit(X[~mask], y[~mask])
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.is_leaf:
            # leaf node
            return np.repeat(self.pred, X.shape[0])
        
        mask = X[:, self.ax] < self.threshold
        left_preds = self.left.transform(X[mask])
        right_preds = self.right.transform(X[~mask])

        preds = np.zeros(X.shape[0])
        preds[mask] = left_preds
        preds[~mask] = right_preds
        return preds
        

class DecisionTreeClassifier(Estimator):

    def __init__(self, max_depth: int = 1, split_rule: str = "gini") -> None:
        super().__init__()
        self.max_depth = max_depth
        if split_rule not in ["gini", "entropy"]:
            raise NotImplementedError("Only support gini impurity or entropy")
        self.split_rule = split_rule
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Estimator:
        self.root = _DecisionNode(self.max_depth, self.split_rule, True)
        self.root.fit(X, y)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.root.transform(X)
    
class DecisionTreeRegressor(Estimator):

    def __init__(self, max_depth: int = 1, split_rule: str = "mse") -> None:
        super().__init__()
        self.max_depth = max_depth
        if split_rule not in ["mse", "mae", "poisson"]:
            raise NotImplementedError("Only support gini impurity or entropy")
        self.split_rule = split_rule
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Estimator:
        self.root = _DecisionNode(self.max_depth, self.split_rule, True)
        self.root.fit(X, y)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.root.transform(X)