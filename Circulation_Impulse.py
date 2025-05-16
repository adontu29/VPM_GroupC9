import numpy as np
import math as m


# ===Linear Impuls===

# This function should be used for one time frame
#

def compute_linear_impulse(X, Y, Z, Wx, Wy, Wz):
    positions = np.stack((X, Y, Z), axis=1)
    strengths = np.stack((Wx, Wy, Wz), axis=1)

    impulses = np.cross(positions, strengths)

    linear_impulse = (0.5 * np.sum(impulses, axis=0)) / np.pi

    return linear_impulse


def compute_angular_impulse(X, Y, Z, Wx, Wy, Wz):
    positions = np.stack((X, Y, Z), axis=1)
    strengths = np.stack((Wx, Wy, Wz), axis=1)

    inner_cross = np.cross(positions, strengths)
    outer_cross = np.cross(positions, inner_cross)

    angular_impulse = (1 / 3) * np.sum(outer_cross, axis=0)

    return angular_impulse


# ===Linear Impulse function, For the comparison====
# This function calculates the Linear Impuls for every time step.
# The input is a array of the Circulation values of the hole vortex ring for every time step
# and the viscousity is constant so can just plug that in the function command.

Viscousity = 0.00013333333333333334


def LinearImpuls(CirculationArray, Viscousity):
    TimeStamps = np.arange(0, 1575, 25) / 1000
    LArray = (2 * np.array(Viscousity) * TimeStamps) ** 0.5
    theta = 1 / LArray

    gamma0_array = np.array(CirculationArray) / (1 - np.exp(-(theta ** 2) / 2))

    R0 = 1
    Impulses = m.pi * gamma0_array * (R0 ** 2)

    return Impulses