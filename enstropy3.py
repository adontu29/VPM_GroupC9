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
time_step_size = 5 * particle_distance ** 2 / ring_strength  # s
n_time_steps = int(20 * ring_radius ** 2 / ring_strength / time_step_size)


@njit(parallel=True)
def calcEnstrophy(x, y, z, wx, wy, wz, radius):
    # Ensure all inputs are 1D arrays and cast to float64
    N = len(x)
    R = radius
    partial_sums = np.zeros(N, dtype=np.float64)  # Change dtype to float64 explicitly

    for i in prange(N):
        local_sum = 0.0

        xi, yi, zi = x[i], y[i], z[i]
        wxi, wyi, wzi = wx[i], wy[i], wz[i]

        for j in range(i + 1, N):
            dx = xi - x[j]
            dy = yi - y[j]
            dz = zi - z[j]
            r2 = dx * dx + dy * dy + dz * dz
            rho2 = r2 / (R * R)

            denom1 = (rho2 + 1.0) ** 3.5
            denom2 = (rho2 + 1.0) ** 4.5

            factor1 = (5.0 - rho2 * (rho2 + 3.5)) / denom1
            factor2 = 3.0 * (rho2 * (rho2 + 4.5) + 3.5) / denom2

            wxj, wyj, wzj = wx[j], wy[j], wz[j]

            dot_w = wxi * wxj + wyi * wyj + wzi * wzj
            dot_r_w1 = dx * wxi + dy * wyi + dz * wzi
            dot_r_w2 = dx * wxj + dy * wyj + dz * wzj

            summand = (factor1 * dot_w + factor2 * dot_r_w1 * dot_r_w2) / (R ** 3)
            local_sum += summand

        partial_sums[i] = local_sum  # Store directly into the array

    return np.sum(partial_sums) / (4.0 * np.pi)


# Read data
DATA_PATH = "dataset"
FILENAME_TEMPLATE = "Vortex_Ring_DNS_Re7500_{:04d}.vtp"
TIMESTAMPS = np.arange(25, 1575, 25)

enstrophies = []
times = []

for stamp in TIMESTAMPS:
    print(stamp)
    filename = f"{DATA_PATH}/{FILENAME_TEMPLATE.format(stamp)}"
    x, y, z, u, v, w, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance(filename)

    vtrInstance = VortexRingInstance(x, y, z, u, v, w, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t)
    x, y, z = vtrInstance.x, vtrInstance.y, vtrInstance.z
    wx, wy, wz = vtrInstance.wx, vtrInstance.wy, vtrInstance.wz
    R = vtrInstance.radius[1]
    times.append(stamp * time_step_size)
    enstrophies.append(calcEnstrophy(x, y, z, wx, wy, wz, R))

plt.plot(times, enstrophies)
plt.show()
