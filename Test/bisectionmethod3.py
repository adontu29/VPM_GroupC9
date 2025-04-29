import sys
import numpy as np
import matplotlib.pyplot as plt
import Test.ReadData as rd
import math
from vtriClass import VortexRingInstance
import vtk


def calcDist(instance1, instance2):
    return np.linalg.norm(np.array(instance1) - np.array(instance2))


# Load dataset
x, y, z, u, v, w, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance(
    "../dataset/Vortex_Ring_DNS_Re7500_0025.vtp"
)

vtrInstance = VortexRingInstance(x, y, z, u, v, w, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t)

# Find min and max coordinates along the x-axis
minx, maxx = np.min(vtrInstance.x), np.max(vtrInstance.x)
minr, maxr = np.min(vtrInstance.rad), np.max(vtrInstance.rad)


def findXPlane(vtrInstance, minx, maxx):
    while maxx - minx > 1e-6:  # Stopping criterion
        middle = (maxx + minx) / 2
        absvort = np.sqrt(np.sqrt(vtrInstance.wy ** 2 + vtrInstance.wz ** 2))
        mask1 = (vtrInstance.x >= minx) & (vtrInstance.x <= middle)
        mask2 = (vtrInstance.x > middle) & (vtrInstance.x <= maxx)
        vort1avg = np.mean(absvort[mask1]) if np.any(mask1) else 0
        vort2avg = np.mean(absvort[mask2]) if np.any(mask2) else 0
        minx, maxx = (minx, middle) if vort1avg > vort2avg else (middle, maxx)
    return middle


def findRad(vtrInstance, minr, maxr):
    while maxr - minr > 1e-6:  # Stopping criterion
        middle = (maxr + minr) / 2
        absvort = np.sqrt(np.sqrt(vtrInstance.wy ** 2 + vtrInstance.wz ** 2))
        mask1 = (vtrInstance.rad >= minr) & (vtrInstance.rad <= middle)
        mask2 = (vtrInstance.rad > middle) & (vtrInstance.rad <= maxr)
        vort1avg = np.mean(absvort[mask1]) if np.any(mask1) else 0
        vort2avg = np.mean(absvort[mask2]) if np.any(mask2) else 0
        minr, maxr = (minr, middle) if vort1avg > vort2avg else (middle, maxr)
    return middle


timeStamps = np.arange(0, 1575, 25)
ringRadius = np.ones(len(timeStamps))
nu = np.ones(len(timeStamps))
ringPos = []
gamma = np.ones(len(timeStamps))
ttab, rtab, xtab = [], [], []

for i, time in enumerate(timeStamps):
    filename = f'../dataset/Vortex_Ring_DNS_Re7500_{str(time).zfill(4)}.vtp'
    try:
        X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance(filename)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        continue

    ringRadius[i], ringPos0 = rd.getRingPosRadius(X, Y, Z, Wx, Wy, Wz)
    vtrInstance = VortexRingInstance(X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t)
    minr, maxr = np.min(vtrInstance.rad), np.max(vtrInstance.rad)
    minx, maxx = np.min(vtrInstance.x), np.max(vtrInstance.x)

    nu[i] = Viscosity[0] if np.ndim(Viscosity) == 1 else Viscosity[1][0]
    ttab.append(time / 1000)
    rtab.append(findRad(vtrInstance, minr, maxr))
    xtab.append(findXPlane(vtrInstance, minx, maxx))
    ringPos.append(np.array(ringPos0))
    gamma[i] = np.sum(np.sqrt(Wx ** 2 + Wy ** 2 + Wz ** 2))
    print(time)

# Compute velocity using NumPy diff
vtab = np.diff(xtab) / 0.025
def regressionM(X, y, M):
    coeffs = np.polyfit(X, y, M)
    return(coeffs)

def function(X, a, b, c):
    X = np.array(X)
    y = a * X ** 2 + b * X + c
    return y

coeffs = regressionM(ttab[:-1], vtab, 2)
a, b, c = coeffs

fig = plt.figure()
ax = plt.axes()

#line, = ax.plot(timeStamps, Velocity, 'b-')
line, = ax.plot(ttab[:-1], function(ttab[:-1],a,b,c), 'r-')


# Plot results
plt.plot(ttab[:-1], vtab)  # Adjust x-axis to match vtab length
plt.xlabel("Time (s)")
plt.ylabel("Velocity")
#plt.ylabel("Radius")
#plt.ylabel("x-position")
plt.show()
