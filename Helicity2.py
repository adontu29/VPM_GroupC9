import numpy as np
import ReadData as rd
from numba import jit
import math

@jit
def my_cross(a, b):
    cx = a[1]*b[2] - a[2]*b[1]
    cy = a[2]*b[0] - a[0]*b[2]
    cz = a[0]*b[1] - a[1]*b[0]
    return np.array([cx, cy, cz])
@jit
def compute_helicity(x, Gamma, sigma):

    N = x.shape[0]
    H_sigma = 0.0

    for p in range(N):
        for q in range(N):
            dx = x[p] - x[q]
            dGamma_cross = my_cross(Gamma[p], Gamma[q])
            r2 = np.dot(dx, dx) + sigma ** 2
            denominator = r2 ** (1.5)
            contribution = np.dot(dx, dGamma_cross) / denominator
            H_sigma += contribution


    H_sigma *= 1 / (4 * math.pi)

    return H_sigma


timeStamps = np.arange(0,1575,25)

ring_center     = np.array([0.0, 0.0, 0.0])   # m, center of the vortex ring
ring_radius     = 1.0               # m, radius of the vortex ring
ring_strength   = 1.0               # m²/s, vortex strength
ring_thickness  = 0.2*ring_radius   # m, thickness of the vortex ring
particle_distance  = 0.25*ring_thickness

timestep = 5 * particle_distance**2/ring_strength
ttab=[]
htab=[]
for i in range(len(timeStamps)):
    stringtime = str(timeStamps[i]).zfill(4)
    # Debugged: Ensure correct file path format
    filename = f'dataset/Vortex_Ring_DNS_Re7500_{stringtime}.vtp'

    try:
        X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance(filename)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        continue
    x = np.stack((X, Y, Z), axis=-1)
    gamma = np.stack((Wx, Wy, Wz), axis=-1)
    H = compute_helicity(x, gamma, sigma=1)
    print(H)
    ttab.append(float(stringtime) * timestep)
    htab.append(H)
