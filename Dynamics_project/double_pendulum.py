"""
Contains class DoublePendulum for modelling movement of double pendulum.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from math import pi
import matplotlib.animation as animation
import time


class NotSolvedError(Exception):
    pass


class NotCreatedError(Exception):
    pass


class DoublePendulum:
    """Initiates instance of double (linked) pendulum

    Initiates system of pendulums with methods to call for derivatives, solve
    and animate. Pendulum system attributes can be set while initiating.
    Animation can be shown or saved to file.
    """

    def __init__(self, L1=1, L2=1, M1=1, M2=1, g=9.81):
        self._L1 = L1
        self._L2 = L2
        self._M1 = M1
        self._M2 = M2
        self._g = g

        self._dt = None

        self._t = None
        self._theta1 = None
        self._theta2 = None

        self._x1 = None
        self._y1 = None
        self._x2 = None
        self._y2 = None

        self._vx1 = None
        self._vy1 = None
        self._vx2 = None
        self._vy2 = None

        self._potential = None
        self._kinetic = None
        self._total = None

        self._animation = None
        self._pendulums = None

    def __call__(self, t, y):
        theta1, omega1, theta2, omega2 = y[0], y[1], y[2], y[3]

        diff_theta1 = omega1
        diff_theta2 = omega2
        del_theta = theta2 - theta1

        diff_omega1 = ((
            + self._M2*self._L1*omega1**2*np.sin(del_theta)*np.cos(del_theta)
            + self._M2*self._g*np.sin(theta2)*np.cos(del_theta)
            + self._M2*self._L2*omega2**2*np.sin(del_theta)
            - (self._M1 + self._M2)*self._g*np.sin(theta1))
            / ((
                self._M1 + self._M2)*self._L1
                - self._M2*self._L1*(np.cos(del_theta))**2))

        diff_omega2 = ((
            - self._M2*self._L2*omega2**2*np.sin(del_theta)*np.cos(del_theta)
            + (self._M1 + self._M2)*self._g*np.sin(theta1)*np.cos(del_theta)
            - (self._M1 + self._M2)*self._L1*omega1**2*np.sin(del_theta)
            - (self._M1 + self._M2)*self._g*np.sin(theta2))
            / ((
                self._M1 + self._M2)*self._L2
                - self._M2*self._L2*(np.cos(del_theta))**2))

        return diff_theta1, diff_omega1, diff_theta2, diff_omega2

    def solve(self, y0, T, dt, angles="rad"):
        if angles == "deg":
            y0 = pi*y0/180

        t = np.linspace(0, T, T/dt)

        solved = scipy.integrate.solve_ivp(fun=self,
                                           t_span=(0, T),
                                           y0=y0,
                                           t_eval=t,
                                           method="Radau")

        self._t = solved.t
        self._dt = dt

        self._theta1 = solved.y[0]
        self._omega1 = solved.y[1]
        self._theta2 = solved.y[2]
        self._omega2 = solved.y[3]

        self._x1 = self._L1*np.sin(self._theta1)
        self._y1 = - self._L1*np.cos(self._theta1)

        self._x2 = self._x1 + self._L2*np.sin(self._theta2)
        self._y2 = self._y1 - self._L2*np.cos(self._theta2)

        self._vx1 = np.gradient(self._x1, self._t)
        self._vy1 = np.gradient(self._y1, self._t)
        self._vx2 = np.gradient(self._x2, self._t)
        self._vy2 = np.gradient(self._y2, self._t)

        self._potential1 = self._M1*self._g*(self._y1 + self._L1)
        self._potential2 = self._M2*self._g*(self._y2 + self._L1 + self._L2)

        self._kinetic1 = (1/2)*self._M1*(self._vx1**2 + self._vy1**2)
        self._kinetic2 = (1/2)*self._M2*(self._vx2**2 + self._vy2**2)

        self._kinetic = (self._kinetic1
                         + self._kinetic2)

        self._potential = (self._potential1
                           + self._potential2)

        self._total = (self._kinetic
                       + self._potential)

    def create_animation(self):
        fig = plt.figure()

        plt.axis("equal")
        plt.axis("off")
        plt.axis((- (self._L1 + self._L2 + 3),
                  (self._L1 + self._L2 + 3),
                  - (self._L1 + self._L2 + 3),
                  (self._L1 + self._L2 + 3)))
        plt.title(
            "Double Pendulum \nM1, M2 = ({} kg, {} kg), L1, L2 = ({} m, {} m)"
            .format(self._M1, self._M2, self._L1, self._L2), weight="bold")

        self._tl = 1500
        self._pendulums, = plt.plot([], [], 'ro-', lw=1.5)
        self._trail1, = plt.plot([], [], "g", lw=1, linestyle="--", alpha=0.5)
        self._trail2, = plt.plot([], [], "y", lw=1, linestyle="--", alpha=0.5)
        self._time = plt.text(
                     -(self._L1+self._L2+2.5),
                     (self._L1+self._L2-1),
                     "Time: {:2.2f} s".format(0),
                     backgroundcolor="k",
                     weight="bold",
                     color="w",
                     rotation="30")

        self._animation = animation.FuncAnimation(
                          fig,
                          self._next_frame,
                          frames=range(1,len(self.x1)+1,10),
                          repeat=None,
                          interval=1000*self.dt,
                          blit=True)

    def _next_frame(self, i):
        self._pendulums.set_data(
                            (0, self.x1[i], self.x2[i]),
                            (0, self.y1[i], self.y2[i]))

        self._trail1.set_data(
                            self.x1[max(0, i-self._tl):i],
                            self.y1[max(0, i-self._tl):i])

        self._trail2.set_data(
                            self.x2[max(0, i-self._tl):i],
                            self.y2[max(0, i-self._tl):i])

        self._time.set_text(
                            "Time: {:2.2f}"
                            .format(i*self._dt))

        return (self._time,
                self._pendulums,
                self._trail1,
                self._trail2)


    def show_animation(self):
        if self._animation is None:
            raise NotCreatedError(
                "Must create animation before call to show")
        else:
            plt.show()

    def save_animation(self,filename="pendulum_in_motion.mp4"):
        if self._animation is None:
            raise NotCreatedError(
                "Must create animation before call to save")
        else:
            self._animation.save(filename, fps=60)

    @property
    def t(self):
        if self._t is None:
            raise NotSolvedError(
                "System must be solved for t to be set")
        else:
            return self._t

    @property
    def dt(self):
        if self._dt is None:
            raise NotSolvedError(
                "System must be solved with intended dt for dt to be set")
        else:
            return self._dt

    @property
    def theta1(self):
        if self._theta1 is None:
            raise NotSolvedError(
                "System must be solved for angles to be calculated")
        else:
            return self._theta1

    @property
    def theta2(self):
        if self._theta2 is None:
            raise NotSolvedError(
                "System must be solved for angles to be calculated")
        else:
            return self._theta2

    @property
    def x1(self):
        if self._x1 is None:
            raise NotSolvedError(
                "System must be solved for x1 coordinates to be calculated")
        else:
            return self._x1

    @property
    def y1(self):
        if self._y1 is None:
            raise NotSolvedError(
                "System must be solved for y1 coordinates to be calculated")
        else:
            return self._y1

    @property
    def x2(self):
        if self._x2 is None:
            raise NotSolvedError(
                "System must be solved for x2 coordinates to be calculated")
        else:
            return self._x2

    @property
    def y2(self):
        if self._y2 is None:
            raise NotSolvedError(
                "System must be solved for y2 coordinates to be calculated")
        else:
            return self._y2

    @property
    def vx1(self):
        if self._vx1 is None:
            raise NotSolvedError(
                "System must be solved for x-dir velocity to be calculated")
        return self._vx1

    @property
    def vy1(self):
        if self._vy1 is None:
            raise NotSolvedError(
                "System must be solved for y-dir velocity to be calculated")
        return self._vy1

    @property
    def vx2(self):
        if self._vx2 is None:
            raise NotSolvedError(
                "System must be solved for x-dir velocity to be calculated")
        return self._vx2

    @property
    def vy2(self):
        if self._vy2 is None:
            raise NotSolvedError(
                "System must be solved for y-dir velocity to be calculated")
        return self._vy2

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

    @property
    def total(self):
        if self._total is None:
            raise NotSolvedError(
                "System must be solved for tot energy to be calculated")
        else:
            return self._total


if __name__ == "__main__":
    dou_pend = DoublePendulum(L1=3, L2=2)

    dou_pend.solve((pi, 0, 40*pi/85, 0), 10, 1/600)

    plt.plot(dou_pend.t, dou_pend.potential)
    plt.plot(dou_pend.t, dou_pend.kinetic)
    plt.plot(dou_pend.t, dou_pend.total)
    plt.xlabel("Time")
    plt.ylabel("Energy (kJ)")
    plt.title("Kin, pot and tot energy with full energy conservation")
    plt.show()

    dou_pend.create_animation()
    dou_pend.save_animation("example_simulation.mp4")
    dou_pend.show_animation()
