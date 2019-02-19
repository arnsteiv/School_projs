"""
Program creates triangle out of chaos game algorithm

Each subexercise is ordered as function for readability and part-by-part 
execution
"""

import numpy as np 
import math as m
import matplotlib.pyplot as plt 

# A) Create three points. Chose to turn it 

def subexerA():
    """Function runs solutions for exercise A"""
    corners = [
            np.array([0,0]), 
            np.array([1,0]), 
            np.array([1/2,m.sqrt(0.75)])]

    plt.scatter(*zip(*corners),
                marker=".",
                color="red",
                alpha=0.8)

    plt.title("Test for equilateral triangle")
    plt.axis("equal")
    plt.show()


# B) Draw 1000 starting points

def subexerB():
    """Function runs solutions for exercise B"""
    N = 1000
    corners = [
            np.array([0,0]), 
            np.array([1,0]), 
            np.array([1/2,m.sqrt(0.75)])]

    startpoints = np.zeros((N,2))

    for i in range(N):
        rands = np.random.random(3)
        rands = rands/np.sum(rands)

        startpoints[i] = (rands[0]*corners[0] 
                        + rands[1]*corners[1] 
                        + rands[2]*corners[2])

    plt.scatter(*zip(*corners),
                marker=".",
                color="red",
                alpha=0.8)

    plt.scatter(*zip(*startpoints),
                color="blue",
                marker="o",
                alpha=0.7)


    plt.axis("equal")
    plt.axis("off")
    plt.title("Testing of whether starting points form triangle")
    plt.show()

# C) Iterating within the n-gon

def subexerC():
    """Function runs solutions for exercise C"""

    corners = [
            np.array([0,0]), 
            np.array([1,0]), 
            np.array([1/2,m.sqrt(0.75)])]

    N = 10000
    points = np.empty((N,2))

    rands = np.random.random(3)
    rands = rands/np.sum(rands)

    x = (rands[0]*corners[0] 
        + rands[1]*corners[1] 
        + rands[2]*corners[2])

    for _ in range(5):
        index = np.random.randint(3)
        corner = corners[index]

        x = (x+corner)/2

    for i in range(N):
        index = np.random.randint(3)
        corner = corners[index]

        x = (x+corner)/2

        points[i] = x

    return points

# D) Plotting the points

def subexerD(points):
    """Function runs solutions for exercise D"""
    corners = [
            np.array([0,0]), 
            np.array([1,0]), 
            np.array([1/2,m.sqrt(0.75)])]

    plt.scatter(*zip(*corners),
                marker=".",
                color="red",
                alpha=0.8)

    plt.scatter(*zip(*points),
                s=0.1,
                alpha=0.8,
                marker=".")

    plt.axis("equal")
    plt.axis("off")
    plt.show()

# E) Adding color

def subexerE():
    """Function runs solutions for exercise E"""

    N = 10000

    corners = [
            np.array([0,0]), 
            np.array([1,0]), 
            np.array([1/2,m.sqrt(0.75)])]

    points = np.empty((N,2))
    colors = np.empty(N)
    rands = np.random.random(3)
    rands = rands/np.sum(rands)

    x = (rands[0]*corners[0] 
        + rands[1]*corners[1] 
        + rands[2]*corners[2])

    for _ in range(5):
        index = np.random.randint(3)
        corner = corners[index]

        x = (x+corner)/2

    for i in range(N):
        index = np.random.randint(3)
        corner = corners[index]
        colors[i] = index
        x = (x+corner)/2

        points[i] = x

    red = points[colors == 0]
    blue = points[colors == 1]
    green = points[colors == 2]

    plt.scatter(*zip(*corners),
                marker=".",
                color="red",
                alpha=0.8)

    plt.scatter(*zip(*red),
                s=0.1,
                alpha=0.8,
                marker=".",
                color="red")

    plt.scatter(*zip(*blue),
                s=0.1,
                alpha=0.8,
                marker=".",
                color="blue")

    plt.scatter(*zip(*green),
                s=0.1,
                alpha=0.8,
                marker=".",
                color="green")

    plt.axis("equal")
    plt.axis("off")
    plt.show()

# F) Alternative colors

def subexerF():
    """Function runs solutions for exercise F"""
    corners = [
            np.array([0,0]), 
            np.array([1,0]), 
            np.array([1/2,m.sqrt(0.75)])]

    N = 10000
    points = np.empty((N,2))
    colors = np.empty((N,3))

    RGBvec = [
            np.array([1,0,0]), 
            np.array([0,1,0]), 
            np.array([0,0,1])]

    rands = np.random.random(3)
    rands = rands/np.sum(rands)

    x = (rands[0]*corners[0] 
        + rands[1]*corners[1] 
        + rands[2]*corners[2])

    c = (rands[0]*RGBvec[0] 
        + rands[1]*RGBvec[1] 
        + rands[2]*RGBvec[2])

    for _ in range(5):
        index = np.random.randint(3)
        corner = corners[index]
        rgb = RGBvec[index]

        x = (x+corner)/2
        c = (c+rgb)/2

    for i in range(N):
        index = np.random.randint(3)
        corner = corners[index]
        rgb = RGBvec[index]

        x = (x+corner)/2
        c = (c+rgb)/2

        points[i] = x
        colors[i] = c

    plt.scatter(*zip(*corners),
                marker=".",
                color="red",
                alpha=0.8)

    plt.scatter(*zip(*points),
                c=colors,s=0.2,
                alpha=0.8,
                marker=".")

    plt.axis("equal")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    subexerA()
    subexerB()
    points = subexerC()
    subexerD(points)
    subexerE()
    subexerF()
