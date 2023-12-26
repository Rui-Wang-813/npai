import numpy as np
from typing import Any

class Variable:
    """Container for a trainable weight variable.

    This is similar to Tensorflow's tf.Variable and pytorch's (deprecated)
    torch.autograd.Variable.
    """

    def __init__(self, value: Any):
        self.value = value
        self.grad = None

def variable(lst: Any):
    if isinstance(lst, list):
        lst = np.array(lst)
    
    return Variable(lst)