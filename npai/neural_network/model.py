"""Neural Network model."""

from .modules import Module
from npai.optimization import Optimizer

import numpy as np
from tqdm import tqdm


def categorical_cross_entropy(pred, labels, epsilon=1e-10):
    """Cross entropy loss function.

    Parameters
    ----------
    pred : np.array
        Softmax label predictions. Should have shape (dim, num_classes).
    labels : np.array
        One-hot true labels. Should have shape (dim, num_classes).
    epsilon : float
        Small constant to add to the log term of cross entropy to help
        with numerical stability.

    Returns
    -------
    float
        Cross entropy loss.
    """
    assert(np.shape(pred) == np.shape(labels))
    return np.mean(-np.sum(labels * np.log(pred + epsilon), axis=1))


def categorical_accuracy(pred, labels):
    """Accuracy statistic.

    Parameters
    ----------
    pred : np.array
        Softmax label predictions. Should have shape (dim, num_classes).
    labels : np.array
        One-hot true labels. Should have shape (dim, num_classes).

    Returns
    -------
    float
        Mean accuracy in this batch.
    """
    assert(np.shape(pred) == np.shape(labels))
    return np.mean(np.argmax(pred, axis=1) == np.argmax(labels, axis=1))


class Sequential:
    """Sequential neural network model.

    Parameters
    ----------
    modules : Module[]
        List of modules; used to grab trainable weights.
    loss : Module
        Final output activation and loss function.
    optimizer : Optimizer
        Optimization policy to use during training.
    """

    def __init__(self, modules, loss=None, optimizer=None):

        for module in modules:
            assert(isinstance(module, Module))
        assert(isinstance(loss, Module))
        assert(isinstance(optimizer, Optimizer))

        self.modules = modules
        self.loss = loss

        self.params = []
        for module in modules:
            self.params += module.trainable_weights

        self.optimizer = optimizer
        self.optimizer.initialize(self.params)

    def forward(self, X, train=True):
        """Model forward pass.

        Parameters
        ----------
        X : np.array
            Input data

        Keyword Args
        ------------
        train : bool
            Indicates whether we are training or testing.

        Returns
        -------
        np.array
            Batch predictions; should have shape (batch, num_classes).
        """
        for module in self.modules:
            X = module.forward(X, train=train)
        return self.loss.forward(X)

    def backward(self, y):
        """Model backwards pass.

        Parameters
        ----------
        y : np.array
            True labels.
        """
        grad = self.loss.backward(y)
        for module in reversed(self.modules):
            grad = module.backward(grad)
        

    def train(self, dataset):
        """Fit model on dataset for a single epoch.

        Parameters
        ----------
        X : np.array
            Input images
        dataset : Dataset
            Training dataset with batches already split.

        Notes
        -----
        You may find tqdm, which creates progress bars, to be helpful:

        Returns
        -------
        (float, float)
            [0] Mean train loss during this epoch.
            [1] Mean train accuracy during this epoch.
        """
        length = 0
        for _, _ in dataset:
            length += 1
        train_loss = []
        train_acc = []

        for X, y in tqdm(dataset, total=length, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', position=1, leave=False):
            pred = self.forward(X, train=True)
            train_loss.append(categorical_cross_entropy(pred, y))
            train_acc.append(categorical_accuracy(pred, y))
            self.backward(y)
            self.optimizer.apply_gradients(self.params)
        
        return np.mean(train_loss), np.mean(train_acc)
        

    def test(self, dataset):
        """Compute test/validation loss for dataset.

        Parameters
        ----------
        dataset : Dataset
            Validation dataset with batches already split.

        Returns
        -------
        (float, float)
            [0] Mean test loss.
            [1] Test accuracy.
        """
        test_loss = []
        test_acc = []

        for X, y in dataset:
            pred = self.forward(X, train=False)
            test_loss.append(categorical_cross_entropy(pred, y))
            test_acc.append(categorical_accuracy(pred, y))
        
        return np.mean(test_loss), np.mean(test_acc)
