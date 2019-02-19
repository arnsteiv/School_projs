"""
Test for Pendulum class.
"""

import nose.tools as nt
import numpy as np
from math import pi, sqrt
from pendulum import NotSolvedError
from pendulum import Pendulum

def test_pendulum():
    """Check if pendulum call gives derivatives."""
    test_pend = Pendulum(L=2.2)
    test_dth, test_dom = test_pend(0, (pi/4, 0.1))

    ex_dth, ex_dom = 0.1, (-9.81/(2.2*sqrt(2)))
    nt.assert_almost_equal(test_dth, ex_dth)
    nt.assert_almost_equal(test_dom, ex_dom)

def test_pendulum_rest():
    """Check if pendulum rests in initial position 0 with velocity 0."""
    test_pend = Pendulum(L=2.2)
    test_dth, test_dom = test_pend(0, (0, 0))

    ex_dth, ex_dom = 0, 0

    nt.assert_equal(test_dth, ex_dth)
    nt.assert_equal(test_dom, ex_dom)

@nt.raises(NotSolvedError)
def test_pendulum_solve_Exceptions():
    """Check if pendulum raises exceptions for non-solved properties."""
    test_pend = Pendulum(L=2.2)
    a = test_pend.t
    b = test_pend.theta
    c = test_pend.omega
    d = test_pend.x
    e = test_pend.y
    f = test_pend.potential
    g = test_pend.vx
    h = test_pend.vy
    i = test_pend.kinetic

def test_pendulum_solve_zero():
    """Check if start in rest ((angle, angular vel) == (0,0)) returns zeros."""
    test_T = 5
    test_dt = 0.1

    test_pend = Pendulum(L=2.2)
    test_pend.solve((0, 0), test_T, test_dt)

    test_ts = np.linspace(0, test_T, test_T/test_dt)
    test_zeros = np.zeros(int(test_T/test_dt))

    np.testing.assert_array_equal(test_pend.t, test_ts)
    np.testing.assert_array_equal(test_pend.theta, test_zeros)
    np.testing.assert_array_equal(test_pend.omega, test_zeros)

def test_pendulum_xy_transformation():
    """Check if motion is transformed to xy-coordinates."""
    test_T = 5
    test_dt = 0.1
    test_L = 2

    test_pend = Pendulum(L=test_L)
    test_pend.solve((pi/2, 0), test_T, test_dt)

    test_L_squared = (test_pend.x)**2 + (test_pend.y)**2
    ex_L_quared = test_L**2

    np.testing.assert_array_almost_equal(test_L_squared, ex_L_quared)

if __name__ == "__main__":
    import nose
    nose.run()
