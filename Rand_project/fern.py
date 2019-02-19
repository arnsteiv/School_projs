import numpy as np
import matplotlib.pyplot as plt


class AffineTransformation():
    """
    A class that creates objects which transforms a vector [x, y],
    by its call function.
    """
    def __init__(self, a = 0, b = 0, c = 0, d = 0, e = 0, f = 0):
        """Initiates Affine transformation function y = Ax + b where:
        
        Input_
        a, b, c, d: params in linear transformation, such that A = [a, b; c, d]
        e, f: params in constant term such that b = [e; f]
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f

    def __call__(self, x, y):
        """Calls affine transformation function vec_y = A*vec_x + b, where:

        Input:
        x, y: Coordinates of input vector vec_x

        Output:
        x, y: Coordinates of output vector vec_y
        """
        return (self.a * x + self.b * y + self.e,
                self.c * x + self.d * y + self.f)


def exer3(save=False):
    """Defines run of exercise 3 as function"""

    # Creating 4 instances of AffineTransformation and putting them in a vector for
    # convenience
    f1 = AffineTransformation(d = 0.16)
    f2 = AffineTransformation(a = 0.85, b = 0.04, c = -0.04, d = 0.85, f = 1.60)
    f3 = AffineTransformation(a = 0.2, b = -0.26, c = 0.23, d = 0.22, f = 1.60)
    f4 = AffineTransformation(a = -0.15, b = 0.28, c = 0.26, d = 0.24, f = 0.44)
    F = [f1, f2, f3, f4]

    # A function that chooses one of the affine transformations according to a
    # probability vector p and transformation function list F
    def prob_vector(p, F):
        p_cumulative = np.cumsum(p)
        r = np.random.random()
        for i in range(len(p_cumulative)):
            if r < p_cumulative[i]:
                return F[i]

    N = 50000
    p = [0.01, 0.85, 0.07, 0.07]
    fern = np.zeros((N, 2))

    # Iterating N-1 times to fill the array with points according to probability
    # vector and functions
    for i in range (N-1):
        function = prob_vector(p, F)
        fern[i+1] = function(fern[i][0], fern[i][1])

    plt.scatter(*zip(*fern), 
                s = 0.1,  
                color = "green", 
                marker = ".")
    plt.axis("equal")
    plt.axis("off")

    if (save==True):
        plt.savefig("barnsley_fern.png", dpi = 300, format = "png")

    plt.show()

if __name__ == "__main__":
    exer3()
