"""
Test for DoublePendulum class.
"""

import nose.tools as nt
import numpy as np
from math import pi, sqrt
from double_pendulum import NotSolvedError
from double_pendulum import DoublePendulum

def test_double_pendulum_rest():
    """Check if double pendulum rests in initial position 0 with velocity 0."""
    test_pend = DoublePendulum(L1=5, L2=5)
    test_dth1, test_dom1, test_dth2, test_dom2 = test_pend(0, (0, 0, 0, 0))

    ex_dth1, ex_dom1 = 0, 0
    ex_dth2, ex_dom2 = 0, 0

    nt.assert_almost_equal(test_dth1, ex_dth1)
    nt.assert_almost_equal(test_dom1, ex_dom1)
    nt.assert_almost_equal(test_dth2, ex_dth2)
    nt.assert_almost_equal(test_dom2, ex_dom1)

@nt.raises(NotSolvedError)
def test_double_pendulum_solve_Exceptions():
    """Check if double pendulum raises exceptions for non-solved properties."""
    test_pend = DoublePendulum(L1=5, L2=5)
    a = test_pend.t
    b = test_pend.theta1
    c = test_pend.theta2
    d1 = test_pend.x1
    e1 = test_pend.y1
    d2 = test_pend.x2
    e2 = test_pend.y2
    f = test_pend.potential
    g1 = test_pend.vx1
    h1 = test_pend.vy1
    g2 = test_pend.vx2
    h2 = test_pend.vy2
    i = test_pend.kinetic
    j = test_pend.total

def test_double_pendulum_solve_zero():
    """Check if start in rest ((angle, angular vel) == (0,0)) returns zeros."""
    test_T = 5
    test_dt = 0.1

    test_pend = DoublePendulum(L1=5, L2=5)
    test_pend.solve((0, 0, 0, 0), test_T, test_dt)

    test_ts = np.linspace(0, test_T, test_T/test_dt)
    test_zeros = np.zeros(int(test_T/test_dt))

    np.testing.assert_array_equal(test_pend.t, test_ts)
    np.testing.assert_array_equal(test_pend.theta1, test_zeros)
    np.testing.assert_array_equal(test_pend.theta2, test_zeros)

def test_dopuble_pendulum_xy_transformation():
    """Check if motion is transformed to xy-coordinates."""
    test_T = 5
    test_dt = 0.1
    test_L1 = 2
    test_L2 = 5

    test_pend = DoublePendulum(L1=test_L1, L2=test_L2)
    test_pend.solve((pi/2, 0, pi, 0), test_T, test_dt)

    test_L1_squared = (test_pend.x1)**2 + (test_pend.y1)**2
    test_L2_squared = ((test_pend.x2 - test_pend.x1)**2
                      + (test_pend.y2 - test_pend.y1)**2)

    ex_L1_quared = test_L1**2
    ex_L2_quared = test_L2**2

    np.testing.assert_array_almost_equal(test_L1_squared, ex_L1_quared)
    np.testing.assert_array_almost_equal(test_L2_squared, ex_L2_quared)

if __name__ == "__main__":
    import nose
    nose.run()
