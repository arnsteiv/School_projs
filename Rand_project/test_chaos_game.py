"""
Unit tests for chaos_game
"""

import nose.tools as nt
import numpy as np
from chaos_game import ChaosGame

def test_ChaosGame_initialization():
    """Test different initialization attributes"""

    testtri = ChaosGame(3, 1/2)
    testquad = ChaosGame(4, 1/3)
    testpent = ChaosGame(5, 1/4)

    nt.assert_almost_equal(testtri.r, 1/2)
    nt.assert_almost_equal(testtri.n, 3)
    nt.assert_almost_equal(testquad.r, 1/3)
    nt.assert_almost_equal(testquad.n, 4)
    nt.assert_almost_equal(testpent.r, 1/4)
    nt.assert_almost_equal(testpent.n, 5)

    nt.assert_almost_equal(testtri.corners[0][0],np.sin(0))
    nt.assert_almost_equal(testtri.corners[0][1],np.cos(0))
    nt.assert_almost_equal(testtri.corners[1][0],np.sin((1/3)*2*np.pi))
    nt.assert_almost_equal(testtri.corners[1][1],np.cos((1/3)*2*np.pi))
    nt.assert_almost_equal(testtri.corners[2][0],np.sin((2/3)*2*np.pi))
    nt.assert_almost_equal(testtri.corners[2][1],np.cos((2/3)*2*np.pi))

@nt.raises(TypeError)
def test_ChaosGame_init_errors1():
    """Test if TypeErrors are raised in initialization"""

    testgon = ChaosGame(0.1, 1/2)
    testgon = ChaosGame("nose", 1/2)
    testgon = ChaosGame([1,2,3], 1/2)

    testgon = ChaosGame(3, "nose")
    testgon = ChaosGame(3, [1,2,3])
    testgon = ChaosGame(3, int(1))

@nt.raises(ValueError)
def test_ChaosGame_init_errors2():
    """Test if ValueErrors are raised in initialization"""

    testgon = ChaosGame(0, 1/2)
    testgon = ChaosGame(1, 1/2)
    testgon = ChaosGame(2, 1/2)
    testgon = ChaosGame(-5, 1/2)

    testgon = ChaosGame(3, float(1))
    testgon = ChaosGame(3, float(2))
    testgon = ChaosGame(3, float(3.5))
    testgon = ChaosGame(3, float(-0.1))

def test_ChaosGame_starting_point():
    """Test if starting points are within radius 1 from origin (unit circle)"""

    for i in range(3,10):
        testgon = ChaosGame(i, 1/2)
        testgon._starting_point()
        nt.assert_less_equal(np.sqrt(testgon.start[0]**2+testgon.start[1]**2),1)


def test_ChaosGame_iterate():
    """Test if iterated points are within radius 1 from origin (unit circle)"""

    testgon = ChaosGame(3,1/2)

    for i in [1000, 3000, 101, 5000]:
        testgon.iterate(i)
        nt.assert_almost_equal(len(testgon.points), i)

    np.testing.assert_array_less(
        np.sqrt(testgon.points[:,0]**2+testgon.points[:,1]**2),1)


if __name__ == "__main__":
    import nose
    nose.run()
