"""
Unit test for ExponentialDecay class.
"""

import nose.tools as nt
from exp_decay import ExponentialDecay

def test_ExponentialDecay():
    """Checking ExponentialDecay call gives derivative."""
    decay_model_test = ExponentialDecay(0.4)
    test = decay_model_test(0, 3.2)
    exact = -1.28
    nt.assert_almost_equal(test, exact)


if __name__ == "__main__":
    import nose
    nose.run()
