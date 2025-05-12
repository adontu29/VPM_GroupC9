import numpy as np
import ReadData2 as rd
from vtriClass import VortexRingInstance
import matplotlib.pyplot as plt
from numba import njit, prange

ring_center = np.array([0.0, 0.0, 0.0])  # m, center of the vortex ring
ring_radius = 1.0  # m, radius of the vortex ring
ring_strength = 1.0  # m²/s, vortex strength
ring_thickness = 0.2 * ring_radius  # m, thickness of the vortex ring

### Particle Distribution Setup
Re = 7500  # Reynolds number
particle_distance = 0.25 * ring_thickness  # m
particle_radius = 0.8 * particle_distance ** 0.5  # m
particle_viscosity = ring_strength / Re  # m²/s, kinematic viscosity
#time_step_size = 5 * particle_distance ** 2 / ring_strength  # s
time_step_size = 0.005808
n_time_steps = int(20 * ring_radius ** 2 / ring_strength / time_step_size)



@njit(parallel=True)
def calcEnstrophy(x, gamma, sigma ):
    N = x.shape[0]
    partial_sums = 0.0
    sigma3 = sigma ** 3
    cutoff2 = (3.0 * sigma) ** 2  # Ignore distant pairs

    for i in prange(N):
        local_sum = 0.0
        for j in range(i+1, N):
            dx0 = x[i, 0] - x[j, 0]
            dx1 = x[i, 1] - x[j, 1]
            dx2 = x[i, 2] - x[j, 2]
            dx = np.array([dx0, dx1, dx2])

            rho2 = (dx0**2 + dx1**2 + dx2**2)/sigma
            if rho2 > cutoff2:
                continue

            dgammap = dx0 * gamma[i, 0] + dx1 * gamma[i, 1] + dx2 * gamma[i, 2]
            dgammaq = dx0 * gamma[j, 0] + dx1 * gamma[j, 1] + dx2 * gamma[j, 2]
            dgamma = dgammap * dgammaq

            denom1 = (rho2 + 1.0) ** 3.5
            denom2 = (rho2 + 1.0) ** 4.5

            dot_gamma = gamma[i, 0] * gamma[j, 0] + gamma[i, 1] * gamma[j, 1] + gamma[i, 2] * gamma[j, 2]

            factor1 = (5.0 - rho2 * (rho2 + 3.5)) / denom1
            factor2 = 3.0 * (rho2 * (rho2 + 4.5) + 3.5) / denom2

            totalfactor = factor1 * dot_gamma + factor2 * dgamma
            finalfactor = totalfactor / sigma3

            local_sum += finalfactor

        partial_sums += local_sum

    return partial_sums / (4.0 * np.pi)



# Read data
DATA_PATH = "dataset2"
FILENAME_TEMPLATE = "Vortex_Ring_{:04d}.vtp"
timeStamps = np.arange(0,8600,200)

ring_center     = np.array([0.0, 0.0, 0.0])   # m, center of the vortex ring
ring_radius     = 1.0               # m, radius of the vortex ring
ring_strength   = 1.0               # m²/s, vortex strength
ring_thickness  = 0.2*ring_radius   # m, thickness of the vortex ring

particle_distance  = 0.3*ring_thickness    # m
particle_radius    = 0.8*particle_distance**0.5
#timestep = 5 * particle_distance**2/ring_strength
timestep = 0.005808

enstrophies = []
times = []


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
    print(calcEnstrophy(x, gamma, sigma=particle_radius))
    enstrophies.append(calcEnstrophy(x, gamma, sigma=particle_radius))
    times.append(float(stringtime) * timestep)
"""for stamp in TIMESTAMPS:
    print(stamp)
    filename = f"{DATA_PATH}/{FILENAME_TEMPLATE.format(stamp)}"
    x, y, z, u, v, w, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance(filename)

    vtrInstance = VortexRingInstance(x, y, z, u, v, w, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t)
    x, y, z = vtrInstance.x, vtrInstance.y, vtrInstance.z
    wx, wy, wz = vtrInstance.wx, vtrInstance.wy, vtrInstance.wz
    R = vtrInstance.radius[1]
    times.append(stamp * time_step_size)
    enstrophies.append(calcEnstrophy(x, gamma, sigma=1))"""

plt.plot(times, enstrophies)
plt.show()
