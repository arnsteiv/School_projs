"""
Contains class Pendulum for modelling pendulum movements.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from math import pi

class NotSolvedError(Exception):
    pass


class Pendulum:
    """Initiates instance of single pendulum

    Initiates pendulum with methods to call for derivatives, solve and animate.
    Pendulum attributes can be set while initiating. Contains properties for
    energy and movement.
    """

    def __init__(self, L=1, M=1, g=9.81):
        self._L = L
        self._M = M
        self._g = g

        self._t = None
        self._theta = None
        self._omega = None
        self._y = None
        self._x = None

        self._potential = None
        self._kinetic = None
        self._vx = None
        self._vy = None

    def __call__(self, t, y):
        theta, omega = y[0], y[1]

        diff_theta = omega
        diff_omega = - (self._g/self._L)*np.sin(theta)

        return diff_theta, diff_omega

    def solve(self, y0, T, dt, angles="rad"):
        if angles == "deg":
            y0 = pi*y0/180

        t = np.linspace(0, T, T/dt)

        solved = scipy.integrate.solve_ivp(fun=self,
                                           t_span=(0, T),
                                           y0=y0,
                                           t_eval=t)

        self._t = solved.t
        self._theta = solved.y[0]
        self._omega = solved.y[1]
        self._x = self._L*np.sin(self._theta)
        self._y = - self._L*np.cos(self._theta)

        self._vx = np.gradient(self._x, self._t)
        self._vy = np.gradient(self._y, self._t)
        self._kinetic = (1/2)*self._M*(self._vx**2 + self._vy**2)
        self._potential = self._M*self._g*(self._y + self._L)

    @property
    def t(self):
        if self._t is None:
            raise NotSolvedError(
                "System must be solved for t to be set")
        else:
            return self._t

    @property
    def theta(self):
        if self._theta is None:
            raise NotSolvedError(
                "System must be solved for angles to be calculated")
        else:
            return self._theta

    @property
    def omega(self):
        if self._omega is None:
            raise NotSolvedError(
                "System must be solved for angular velocities to be calculated")
        else:
            return self._omega

    @property
    def x(self):
        if self._x is None:
            raise NotSolvedError(
                "System must be solved for x coordinates to be calculated")
        else:
            return self._x

    @property
    def y(self):
        if self._y is None:
            raise NotSolvedError(
                "System must be solved for y coordinates to be calculated")
        else:
            return self._y

    @property
    def vx(self):
        if self._vx is None:
            raise NotSolvedError(
                "System must be solved for x-dir velocity to be calculated")
        return self._vx

    @property
    def vy(self):
        if self._vy is None:
            raise NotSolvedError(
                "System must be solved for y-dir velocity to be calculated")
        return self._vy

    @property
    def potential(self):
        if self._potential is None:
            raise NotSolvedError(
                "System must be solved for pot energy to be calculated")
        else:
            return self._potential

    @property
    def kinetic(self):
        if self._kinetic is None:
            raise NotSolvedError(
                "System must be solved for kin energy to be calculated")
        else:
            return self._kinetic


class DampenedPendulum(Pendulum):
    """Initiates instance of dampened pendulum

    Initiated with (length L, mass M, dampening factor B, gravity constant g)
    Called with (time t, value y) gives the angular velocity (derivative of
    angle) and angular acceleration (derivative of ang velocity).
    """
    def __init__(self, L=1, M=1, B=0.1, g=9.81):
        super().__init__(L, M, g)
        self._B = B

    def __call__(self, t, y):
        theta, omega = y[0], y[1]

        diff_theta = omega
        diff_omega = (
            - (self._g/self._L)*np.sin(theta)
            - (self._B/self._M)*omega)

        return diff_theta, diff_omega

if __name__ == "__main__":
    pend = Pendulum(L=5)
    pend.solve((pi/2, 0.5), 20, 0.00001)

    dpend = DampenedPendulum(L=5)
    dpend.solve((pi/2, 0.5), 20, 0.00001)

    plt.plot(pend.t, pend.theta)
    plt.title("Angle of pendulum with full energy conservation")
    plt.xlabel("Time")
    plt.ylabel("Angle from equilibrium")
    plt.show()

    plt.plot(pend.t, pend.potential)
    plt.plot(pend.t, pend.kinetic)
    plt.plot(pend.t, (pend.kinetic+pend.potential))
    plt.xlabel("Time")
    plt.ylabel("Energy (kJ)")
    plt.title("Kin, pot and tot energy with full energy conservation")
    plt.show()

    plt.plot(dpend.t, (dpend.kinetic + dpend.potential))
    plt.xlabel("Time")
    plt.ylabel("Energy (kJ)")
    plt.title("Total energy for dampened pendulum")

    plt.show()
