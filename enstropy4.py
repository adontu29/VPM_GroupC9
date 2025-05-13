from numba import njit, prange
import numpy as np
import ReadData2 as rd

@njit
def compute_cell_index(pos, cell_size, grid_min):
    return ((pos - grid_min) // cell_size).astype(np.int32)

@njit
def get_neighboring_cells(cell_idx):
    neighbors = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                neighbors.append((cell_idx[0] + dx, cell_idx[1] + dy, cell_idx[2] + dz))
    return neighbors

@njit(parallel=True)
def calcEnstrophy_CellList(x, gamma, sigma):
    N = x.shape[0]
    sigma3 = sigma**3
    cutoff2 = (3.0 * sigma) ** 2
    cell_size = 2.5 * sigma

    # Grid bounds
    grid_min = np.min(x, axis=0) - 1e-5
    grid_max = np.max(x, axis=0) + 1e-5
    grid_dims = np.ceil((grid_max - grid_min) / cell_size).astype(np.int32)

    # Build grid: map cell index → particle indices
    cell_particles = dict()
    for i in range(N):
        cell_idx = tuple(compute_cell_index(x[i], cell_size, grid_min))
        if cell_idx in cell_particles:
            cell_particles[cell_idx].append(i)
        else:
            cell_particles[cell_idx] = [i]

    # Compute enstrophy
    enstrophy_sum = 0.0
    for i in prange(N):
        xi = x[i]
        gi = gamma[i]
        cell_idx = tuple(compute_cell_index(xi, cell_size, grid_min))
        neighbors = get_neighboring_cells(cell_idx)

        local_sum = 0.0
        for ncell in neighbors:
            if ncell not in cell_particles:
                continue
            for j in cell_particles[ncell]:
                dx = xi - x[j]
                rho2 = dx[0]**2 + dx[1]**2 + dx[2]**2
                if rho2 > cutoff2:
                    continue

                gj = gamma[j]
                dgammap = dx[0]*gi[0] + dx[1]*gi[1] + dx[2]*gi[2]
                dgammaq = dx[0]*gj[0] + dx[1]*gj[1] + dx[2]*gj[2]
                dgamma = dgammap * dgammaq

                denom1 = (rho2 + 1.0)**3.5
                denom2 = (rho2 + 1.0)**4.5

                dot_gamma = gi[0]*gj[0] + gi[1]*gj[1] + gi[2]*gj[2]

                factor1 = (5.0 - rho2 * (rho2 + 3.5)) / denom1
                factor2 = 3.0 * (rho2 * (rho2 + 4.5) + 3.5) / denom2

                totalfactor = (factor1 * dot_gamma + factor2) * dgamma
                local_sum += totalfactor / sigma3

        enstrophy_sum += local_sum

    return enstrophy_sum / (4.0 * np.pi)


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
    print(calcEnstrophy_CellList(x, gamma, sigma=1))
    enstrophies.append(calcEnstrophy_CellList(x, gamma, sigma=1))
    times.append(float(stringtime) * time_step_size)
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