import numpy as np


class ConstantNet:
    """Test double: always returns a fixed motor output, regardless of input."""

    def __init__(self, output, n_params=3):
        self.output = np.array(output, dtype=float)
        self.params = np.zeros(n_params)

    def activate(self, inputs):
        return self.output

    def set_params(self, params):
        self.params = np.asarray(params, dtype=float)
