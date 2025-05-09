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
def calcEnstrophy(x, Gamma, sigma):
    # Ensure all inputs are 1D arrays and cast to float64
    N = x.shape[0]
    #R = radius
    partial_sums = np.zeros(N, dtype=np.float64)  # Change dtype to float64 explicitly
    local_sum = np.zeros(N, dtype=np.float64)
    for i in prange(N):

        for j in range(N):
            dx = x[i] - x[j]
            dgammap = np.dot(dx, gamma[i])
            dgammaq = np.dot(dx, gamma[j])
            dgamma = dgammap * dgammaq
            rho = np.abs(dx) / sigma
            rho2 = rho**2

            denom1 = (rho2 + 1.0) ** 3.5
            denom2 = (rho2 + 1.0) ** 4.5

            factor1 = (5.0 - rho2 * (rho2 + 3.5)) / denom1
            factor2 = 3.0 * (rho2 * (rho2 + 4.5) + 3.5) / denom2
            totalfactor = (factor1 * np.dot(gamma[i], gamma[j]) + factor2) * dgamma
            finalfactor = totalfactor / (sigma**3)

            local_sum += finalfactor

        partial_sums[i] = local_sum  # Store directly into the array

    return np.sum(partial_sums) / (4.0 * np.pi)


# Read data
DATA_PATH = "dataset"
FILENAME_TEMPLATE = "Vortex_Ring_DNS_Re7500_{:04d}.vtp"
timeStamps = np.arange(25, 1575, 25)

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
    enstrophies.append(calcEnstrophy(x, gamma, sigma=1))
    times.append(stringtime * time_step_size)
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
