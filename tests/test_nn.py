import numpy as np
import pytest

from alife.nn import FeedForwardNet


def test_param_count_matches_pybrain_buildnetwork_formula():
    # (n_in + 1) * n_hidden + (n_hidden + 1) * n_out, matching
    # pybrain.tools.shortcuts.buildNetwork(8, hid, 2) defaults (bias=True).
    for hid, expected in [(4, 46), (8, 90), (16, 178)]:
        net = FeedForwardNet(8, hid, 2)
        assert net.n_params == expected


def test_set_params_round_trip():
    net = FeedForwardNet(8, 8, 2)
    params = np.arange(net.n_params, dtype=float)
    net.set_params(params)
    assert np.array_equal(net.params, params)


def test_set_params_rejects_wrong_shape():
    net = FeedForwardNet(8, 8, 2)
    with pytest.raises(ValueError):
        net.set_params(np.zeros(net.n_params - 1))


def test_activate_output_shape():
    net = FeedForwardNet(8, 8, 2)
    out = net.activate(np.zeros(8))
    assert out.shape == (2,)


def test_activate_zero_params_gives_zero_output():
    net = FeedForwardNet(8, 8, 2, params=np.zeros(90))
    out = net.activate(np.random.uniform(-1, 1, 8))
    assert np.allclose(out, 0.0)


def test_activate_is_deterministic_given_fixed_params():
    net = FeedForwardNet(8, 8, 2, params=np.linspace(-1, 1, 90))
    inputs = np.linspace(-1, 1, 8)
    out1 = net.activate(inputs)
    out2 = net.activate(inputs)
    assert np.array_equal(out1, out2)
