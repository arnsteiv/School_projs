"""
Contains class ExponentialDecay, for use in modelling pendulum movements.
"""

import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt


class ExponentialDecay:
    """Model for creating instances of exponential decay

    Creates simple model of system in exponential decay. Initializes with core
    constant, can be called and solved.
    """

    def __init__(self, a):
        self._a = a

    def __call__(self, t, u):
        return -self._a*u

    def solve(self, u0, T, dt):
        t = np.linspace(0, T, T/dt)
        solved = scipy.integrate.solve_ivp(fun=self,
                                           t_span=(0, T),
                                           y0=(u0,),
                                           t_eval=t)
        return solved.t, solved.y[0]


if __name__ == "__main__":

    decay_model = ExponentialDecay(0.4)

    for i in range(1, 51, 10):
        t, u = decay_model.solve(i, 10, 0.1)

        plt.plot(t, u)

    plt.xlabel("t")
    plt.ylabel("Modelled decay, u(t)")
    plt.title("Model of exponential decay")

    plt.show()
