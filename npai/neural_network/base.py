"""Base classes for layer/activation abstractions.

You can use this file as a reference for the architecture of this program;
do not modify this file.
"""

class Module:
    """Base class for NumpyNet network layers and activation functions.

    NOTE: students are strongly discouraged from modifying this class.

    Attributes
    ----------
    trainable_weights : Variable[]
        List of variables that can be trained in this module.
    """
    def __init__(self):
        self.trainable_weights = []

    def forward(self, x, train=True):
        """Forward propagation.

        Parameters
        ----------
        x : np.array
            Input for this layer, x_{k-1}.

        Keyword Args
        ------------
        train : bool
            Indicates whether we are in training or validation/testing.

        Returns
        -------
        np.array
            Output of this layer x_k = f_k(w_k, x_{k-1}) for weights w_k.
        """
        raise NotImplementedError()

    def backward(self, grad):
        """Backward propagation.

        Should set ```self.trainable_weights[*].grad``` to the mean batch
        gradients for the trainable weights in this layer,

            E[dL/dw_k] = E[(dx_k/dw_k)^T dL/dx_k],

        and return the gradients flowing to the previous layer,

            dL/dx_{k-1} = (dx_k/dx_{k-1})^T dL/dx_k.

        Parameters
        ----------
        grad : np.array
            Gradient flowing backwards from the next layer, dL/dx_k.

        Returns
        -------
        np.array
            Gradients for the inputs to this layer, dL/dx_{k-1}.
        """
        raise NotImplementedError()



