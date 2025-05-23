
import numpy as np
import ReadData as rd
from vtriClass import VortexRingInstance
import matplotlib.pyplot as plt
from numba import njit
from scipy.spatial import cKDTree
import math

# --------------------------
# Vortex ring configuration
# --------------------------
ring_center = np.array([0.0, 0.0, 0.0])  # m
ring_radius = 1.0  # m
ring_strength = 1.0  # m²/s
ring_thickness = 0.2 * ring_radius  # m

# Particle configuration
Re = 7500
particle_distance = 0.22 * ring_thickness  # m
particle_radius = 0.8 * particle_distance ** 0.5
particle_viscosity = ring_strength / Re
# time_step_size     = 3 * particle_distance**2 / ring_strength  # s
time_step_size = 0.005808
n_time_steps = int(20 * ring_radius ** 2 / ring_strength / time_step_size)


# ------------------------------------------
# Numba-safe enstrophy calculation function
# ----
#
# --------------------------------------
@njit
def calcEnstrophy_vec_numba(x, y, z, wx, wy, wz, particle_radius):
    N = x.shape[0]
    enstrophy = 0.0
    radius_cubed = particle_radius ** 3

    for i in range(N):

            for j in range(N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dz = z[i] - z[j]

                r2 = dx * dx + dy * dy + dz * dz
                rho2 = r2 / (particle_radius)

                denom1 = (rho2 + 1.0) ** 3.5
                denom2 = (rho2 + 1.0) ** 4.5

                factor1 = (5.0 - rho2 * (rho2 + 3.5)) / denom1
                factor2 = 3.0 * (rho2 * (rho2 + 4.5) + 3.5) / denom2

                # Vorticity components
                wxi, wyi, wzi = wx[i], wy[i], wz[i]
                wxj, wyj, wzj = wx[j], wy[j], wz[j]

                # Dot products
                dot_strength = wxi * wxj + wyi * wyj + wzi * wzj
                dot_diff_strength1 = dx * wxi + dy * wyi + dz * wzi
                dot_diff_strength2 = dx * wxj + dy * wyj + dz * wzj

                # Scalar summand
                scalar_summand = (
                                             factor1 * dot_strength + factor2 * dot_diff_strength1 * dot_diff_strength2) / radius_cubed
                enstrophy += scalar_summand

    return enstrophy / (4.0 * np.pi)


# -------------------------------
# File and time step configuration
# -------------------------------
DATA_PATH = "dataset2"
FILENAME_TEMPLATE = "Vortex_Ring_{:04d}.vtp"
TIMESTAMPS = np.arange(25, 8600, 250)  # Process every 10th file (10x speedup)
TIMESTAMPS = np.arange(25, 8600, 500)  # Process every 10th file (10x speedup)

enstrophies = []
times = []
cumulative_time = 0.0

# -------------------------------
# Main processing loop
# -------------------------------
for stamp in TIMESTAMPS:
    print(f"Processing timestep {stamp}")
    filename = f"{DATA_PATH}/{FILENAME_TEMPLATE.format(stamp)}"
    x, y, z, u, v, w, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance(filename)
    vtrInstance = VortexRingInstance(x, y, z, u, v, w, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t)

    # Ensure all arrays are float64 for Numba
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    Wx = np.asarray(Wx, dtype=np.float64)
    Wy = np.asarray(Wy, dtype=np.float64)
    Wz = np.asarray(Wz, dtype=np.float64)

    # Estimate particle distance
    positions = np.stack((x, y, z), axis=1)
    tree = cKDTree(positions)
    dists, _ = tree.query(positions, k=2)
    particle_distance = np.mean(dists[:, 1])  # skip self-distance

    # Estimate ring strength
    strengths = np.stack((Wx, Wy, Wz), axis=1)
    ring_strength = np.mean(np.linalg.norm(strengths, axis=1))

    # Time step
    # time_step_size = 5 * particle_distance**2 / ring_strength
    time_step_size = 0.005808
    cumulative_time = time_step_size * stamp
    radius_val = Radius[int(stamp / 25)].item()
    print(radius_val)
    print(Radius[0])
    print(calcEnstrophy_vec_numba(x, y, z, Wx, Wy, Wz, float(Radius[0].item())))
    print(calcEnstrophy_vec_numba(x, y, z, Wx, Wy, Wz, radius_val))
    # Enstrophy calculation
    enstrophy = calcEnstrophy_vec_numba(x, y, z, Wx, Wy, Wz, particle_radius)
    #print(stamp)
    enstrophy = calcEnstrophy_vec_numba(x, y, z, Wx, Wy, Wz, radius_val)
    enstrophies.append(enstrophy / (0.2 * 4 * math.pi * math.pi))
    times.append(cumulative_time)
# -------------------------------
# Plot results
# -------------------------------
plt.plot(times, enstrophies)
plt.xlabel("Time (s)")
plt.ylabel("Enstrophy")
plt.title("Enstrophy Evolution with Dynamic Time Step (Reduced Data)")
plt.grid(True)
plt.tight_layout()
plt.show()
