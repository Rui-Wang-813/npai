class Optimizer:
    """Optimization policy base class."""

    def initialize(self, params):
        """Initialize optimizer state.

        Parameters
        ----------
        params : Variable[]
            List of parameters to initialize state for.
        """
        # Can leave this blank if the optimizer has no state, i.e. SGD.
        pass

    def apply_gradients(self, params):
        """Apply gradients to parameters.

        Parameters
        ----------
        params : Variable[]
            List of parameters that the gradients correspond to.
        """
        raise NotImplementedError()