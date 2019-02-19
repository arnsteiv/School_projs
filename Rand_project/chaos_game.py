"""
Program with class for chaos game with n-gon.

Each subexercise is ordered as function for readability and part-by-part
execution
"""

import numpy as np
import matplotlib.pyplot as plt

class ChaosGame:
    """Class for modelling chaos game"""

    def __init__(self, n, r=1/2):
        """Initializes ChaosGame object

        Input:
        n: rank of polygon ("n-gon"). Must be integer
        r: Distance between corner and point traversed when iterating in
        chaos game. Must be decimal between 0 and 1.

        Also initates RGB-vector for later use, generates n-gon with n corners.
        """
        if not isinstance(n, int):
            raise TypeError(
                "n must be integer, is {}".format(type(n)))

        if not isinstance(r, float):
            raise TypeError(
                "r must be decimal (float), is {}".format(type(r)))

        if not (r > 0 and r <1 ):
            raise ValueError(
                "r out of range, must be decimal",
                "between 0 and 1, is {}".format(r))

        if (n < 3):
            raise ValueError(
                "n out of range, must be 3 or above, is {}".format(n))

        self.n = n
        self.r = r
        self.RGBvec = [
                    np.array([1,0,0]),
                    np.array([0,1,0]),
                    np.array([0,0,1])]

        self._generate_ngon()
        self.points = None

    def _generate_ngon(self):
        """Generates n-gon with n corners in object."""
        self.corners = []

        for i in range(self.n):

            angle = (i/self.n)*2*np.pi

            self.corners.append(np.array([np.sin(angle), np.cos(angle)]))

    def _starting_point(self):
        """Generates starting point for iteration in n-gon"""

        # Generating an array of n floats which sums to 1
        rands = np.random.random(self.n)
        rands = rands/np.sum(rands)

        x = np.zeros(2)

        # Multiplying the probability vector with the corners
        for i in range(self.n):
            x += rands[i]*self.corners[i]

        # Generating color starting point
        colrands = np.random.random(3)
        colrands = colrands/np.sum(colrands)

        col = (colrands[0]*self.RGBvec[0]
            + colrands[1]*self.RGBvec[1]
            + colrands[2]*self.RGBvec[2])

        # Setting internal variables
        self.start = x
        self.colorstart = col

        # Return for test given in assignment
        return self.start

    def iterate(self, steps, discard=5):
        """Creates N=[steps] points in n-gon by iteration

        Input:
        steps: Number of iterations/points
        discard (defaults to 5): number of iterations to be discarded before
        recorded iteration starts

        Also colormap for use with .show()
        """
        self._starting_point()
        self.points = np.zeros((steps,2))
        self.colors = np.zeros((steps,3))

        x = self.start
        c = self.colorstart

        for _ in range(discard):
            index = np.random.randint(self.n)

            x = self.r*x + (1-self.r)*self.corners[index]
            c = self.r*c + (1-self.r)*self.RGBvec[index % 3]

        for i in range(steps):
            index = np.random.randint(self.n)

            x = self.r*x + (1-self.r)*self.corners[index]
            c = self.r*c + (1-self.r)*self.RGBvec[index % 3]

            self.points[i] = x
            self.colors[i] = c

    def plot_ngon(self):
        """Plots ngon corners w/o iterated points."""
        plt.scatter(*zip(*self.corners),
                    marker=".",
                    color="red",
                    alpha=0.8)

        plt.axis("equal")
        plt.axis("off")
        plt.show()


    def show(self, colorize=True):
        """Plots iterated points. If there are no points to be plotted,
        self.iterate() is called with 10000 steps and otherwise default
        values.

        Input:
        colorize (defaults to true): Colorizes points with existing cmap."""

        if not np.sum(self.points):
            self.iterate(10000)
            print("Iterate() with 10000 steps has been called, call iterate() " \
                + "first to customize ")

        if colorize:
            plt.scatter(*zip(*self.points),
                        c=self.colors,
                        s=0.2,
                        alpha=0.8,
                        marker=".")

        else:
            plt.scatter(*zip(*self.points),
                        color="black",
                        s=0.2,
                        alpha=0.8,
                        marker=".")

        plt.axis("equal")
        plt.axis("off")
        plt.show()


    def savepng(self, outfile, colorize=True):
        """ Saves plot of iterated points as png file

        Input:
        outfile: Name of outfile. Filetype must be unspecified or .png
        colorize (defaults to True): Option to create plot with colors
        """

        if not np.sum(self.points):
            self.iterate()

        if outfile[-4:] == ".png":
            pass
        elif "." not in outfile:
            outfile += ".png"
        else:
            raise NameError(
                "Outfile filetype must be non-specified",
                "or .png, is {}".format(outfile))

        if colorize:
            plt.scatter(*zip(*self.points),
                        c=self.colors,
                        s=0.2,
                        alpha=0.8,
                        marker=".")

        else:
            plt.scatter(*zip(*self.points),
                        color="black",
                        s=0.2,
                        alpha=0.8,
                        marker=".")

        plt.axis("equal")
        plt.axis('off')

        plt.savefig(outfile, dpi=300, transparent=True)
        plt.clf()

def subexerB():
    """Function runs solutions for exercise B"""
    for i in range(3,9):
        ngon = ChaosGame(i)

        ngon.plot_ngon()

def subexerC():
    """Function runs solutions for exercise C"""
    testgon = ChaosGame(3)

    testlist = np.zeros((1000,2))

    for i in range(1000):
        testlist[i] = testgon._starting_point()

    plt.scatter(*zip(*testgon.corners),
                marker=".",
                color="red",
                alpha=0.8)

    plt.scatter(*zip(*testlist),
                marker=".",
                color="blue",
                alpha=0.5)

    plt.axis("equal")
    plt.axis("off")
    plt.show()

def subexerE():
    """Function runs solutions for exercise E"""
    testgon2 = ChaosGame(5,1/2)

    testgon2.iterate(100000)
    testgon2.show()

def subexerG():
    """Function runs solutions for exercise G"""
    testgon3 = ChaosGame(5,1/4)

    testgon3.iterate(10000)
    testgon3.savepng("testgon1")

    testgon3 = ChaosGame(3,1/2)

    testgon3.iterate(10000)
    testgon3.savepng("testgon2.png")


def subexerI():
    """Function runs solutions for exercise I"""
    i = 1
    for n, r in zip([3, 4, 5, 5, 6], [1/2, 1/3, 1/3, 3/8, 1/3]):
        finalgon = ChaosGame(n, r)

        finalgon.iterate(1000)
        finalgon.savepng("chaos{}.png".format(i))
        i += 1

if __name__ == "__main__":
    subexerB()
    subexerC()
    subexerE()
    subexerG()
    subexerI()

    pass
