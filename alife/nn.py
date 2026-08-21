"""Minimal numpy-only feedforward net, replacing the pybrain dependency.

Architecture mirrors pybrain.tools.shortcuts.buildNetwork(n_in, n_hidden,
n_out) with its defaults (bias=True, hiddenclass=SigmoidLayer,
outclass=LinearLayer): a bias unit feeding a sigmoid hidden layer, feeding
a linear output layer. Matching that architecture keeps the evolved
parameter-count math (used to size mutation noise) unchanged from before
the port.
"""

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class FeedForwardNet:
    def __init__(self, n_in, n_hidden, n_out, params=None):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.n_params = (n_in + 1) * n_hidden + (n_hidden + 1) * n_out
        if params is None:
            params = np.random.uniform(-1, 1, self.n_params)
        self.set_params(params)

    def set_params(self, params):
        params = np.asarray(params, dtype=float)
        if params.shape != (self.n_params,):
            raise ValueError(
                "expected %d params, got shape %s" % (self.n_params, params.shape)
            )
        self.params = params
        i = 0
        n_in, n_hidden, n_out = self.n_in, self.n_hidden, self.n_out
        self.w_in_hidden = params[i:i + n_in * n_hidden].reshape(n_in, n_hidden)
        i += n_in * n_hidden
        self.b_hidden = params[i:i + n_hidden]
        i += n_hidden
        self.w_hidden_out = params[i:i + n_hidden * n_out].reshape(n_hidden, n_out)
        i += n_hidden * n_out
        self.b_out = params[i:i + n_out]

    def activate(self, inputs):
        inputs = np.asarray(inputs, dtype=float)
        hidden = _sigmoid(inputs @ self.w_in_hidden + self.b_hidden)
        return hidden @ self.w_hidden_out + self.b_out
