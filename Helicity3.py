import numpy as np
import ReadData as rd
from numba import jit
import math
from matplotlib import pyplot as plt
from numba import njit, prange

@njit(parallel=True)
def getHelicity(x, Gamma, sigma):
    N = x.shape[0]
    H_sigma = 0.0

    for p in prange(N):  # Outer loop parallelized
        for q in range(N):
            dx0 = x[p,0] - x[q,0]
            dx1 = x[p,1] - x[q,1]
            dx2 = x[p,2] - x[q,2]

            cx = Gamma[p,1]*Gamma[q,2] - Gamma[p,2]*Gamma[q,1]
            cy = Gamma[p,2]*Gamma[q,0] - Gamma[p,0]*Gamma[q,2]
            cz = Gamma[p,0]*Gamma[q,1] - Gamma[p,1]*Gamma[q,0]

            r2 = dx0**2 + dx1**2 + dx2**2 + sigma**2
            denominator = r2 ** 1.5

            contribution = (dx0 * cx + dx1 * cy + dx2 * cz) / denominator
            H_sigma += contribution

    H_sigma *= 1 / (4 * np.pi)
    return H_sigma







timeStamps = np.arange(0,8600,25)

ring_center     = np.array([0.0, 0.0, 0.0])   # m, center of the vortex ring
ring_radius     = 1.0               # m, radius of the vortex ring
ring_strength   = 1.0               # m²/s, vortex strength
ring_thickness  = 0.2*ring_radius   # m, thickness of the vortex ring
particle_distance  = 0.25*ring_thickness

#timestep = 5 * particle_distance**2/ring_strength
timestep = 0.005808
ttab=[]
htab=[]
for i in range(len(timeStamps)):
    stringtime = str(timeStamps[i]).zfill(4)
    # Debugged: Ensure correct file path format
    filename = f'dataset2/Vortex_Ring_{stringtime}.vtp'

    try:
        X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance(filename)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        continue
    x = np.stack((X, Y, Z), axis=-1)
    gamma = np.stack((Wx, Wy, Wz), axis=-1)
    H = getHelicity(x, gamma, sigma=1)
    print(H)
    ttab.append(float(stringtime) * timestep)
    htab.append(H)
plt.plot(ttab, htab)
plt.ylabel('Helicity (|H|)')
plt.xlabel('Time')
plt.title('Helicity vs Time')
plt.grid(True, which="both", ls="--")
plt.ylim(-1, 1)
plt.show()