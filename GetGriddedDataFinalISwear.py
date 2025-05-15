import numpy as np
import math as m
import matplotlib.pyplot as plt
import ReadData as rd
from numba import njit, prange

# ============================ Constants and Parameters ============================

ring_center   = np.array([0.0, 0.0, 0.0])
ring_radius   = 1.0
ring_strength = 1.0
ring_thickness = 0.2 * ring_radius

Re = 750
particle_distance  = 0.22 * ring_thickness
particle_radius    = 0.8 * particle_distance**0.5
particle_viscosity = ring_strength / Re
time_step_size     = 25 * 3 * particle_distance**2 / ring_strength
n_time_steps       = int(100 * ring_radius**2 / ring_strength / time_step_size)
max_timesteps = 8600
no_timesteps = int(8600 / 25 + 1)

timeStampsNames = np.arange(0, max_timesteps + 1, 25)
timeStamps = np.arange(0, no_timesteps * time_step_size, time_step_size)

# ============================ Numba-accelerated Functions ============================

print(timeStamps[300])
@njit(fastmath=True)
def reg_zeta(rho, radius):
    rho = rho / radius
    return 15.0 / (8.0 * np.pi * (rho**2 + 1.0)**3.5) / radius**3

@njit(fastmath=True)
def reg_q(rho, radius):
    rho = rho / radius
    return (rho**3 * (rho**2 + 2.5)) / ((rho**2 + 1.0)**2.5) / (4.0 * np.pi)

@njit(parallel=True, fastmath=True)
def compute_vorticity_numba(grid_points, particle_pos, Gamma, particle_radius):
    M = grid_points.shape[0]
    N = particle_pos.shape[0]
    omega_grid = np.zeros((M, 3), dtype=np.float64)

    for p in prange(N):
        r = grid_points - particle_pos[p]      # (M, 3)
        r_mag = np.sqrt(np.sum(r**2, axis=1))
        r_mag[r_mag == 0] = 1e-12

        r_squared = r_mag**2
        r_cubed = r_mag**3
        dot_rg = np.sum(r * Gamma[p], axis=1)

        for i in range(M):
            zeta_val = reg_zeta(r_mag[i], particle_radius)
            q_val = reg_q(r_mag[i], particle_radius)

            term1 = (zeta_val - q_val / r_cubed[i]) * Gamma[p]
            scalar = (3 * q_val / r_cubed[i] - zeta_val) * (dot_rg[i] / r_squared[i])
            term2 = scalar * r[i]

            omega_grid[i] += term1 + term2

    return omega_grid

# ============================ Vorticity Computation Wrapper ============================

def compute_vorticity_field(grid_points, particle_pos, Gamma, particle_radius, XGrid, YGrid, ZGrid, ringPos,time):
    """
    Computes both numerical and analytical vorticity fields.
    """
    Y_shifted = YGrid - ringPos[1]
    Z_shifted = ZGrid - ringPos[2]
    X_shifted = XGrid - ringPos[0]

    r_yz = np.sqrt(Y_shifted ** 2 + Z_shifted ** 2)
    radial_distance_to_core = np.abs(r_yz - ring_radius)
    theta = np.arctan2(Z_shifted, Y_shifted)

    ring_thickness_t = np.sqrt(4 * particle_viscosity * time + ring_thickness ** 2)
    vorticity_magnitude = (ring_strength / (np.pi * ring_thickness_t ** 2)) * \
                          np.exp(-radial_distance_to_core ** 2 / ring_thickness_t ** 2)
    x_decay = np.exp(-X_shifted ** 2 / ring_thickness_t ** 2)

    omega_x_analytic = np.zeros_like(XGrid)
    omega_y_analytic = -vorticity_magnitude * np.sin(theta) * x_decay
    omega_z_analytic = vorticity_magnitude * np.cos(theta) * x_decay

    omega_grid_analytical = np.stack((
        omega_x_analytic.ravel(),
        omega_y_analytic.ravel(),
        omega_z_analytic.ravel()
    ), axis=-1)

    omega_grid = compute_vorticity_numba(grid_points, particle_pos, Gamma, particle_radius)
    return omega_grid, omega_grid_analytical

# ============================ Grid Setup ============================

xGrid = np.arange(-0.3, 9.5, 0.1)
yGrid = np.arange(-1.5, 1.5, 0.05)
zGrid = np.arange(-1.5, 1.5, 0.05)

XGrid, YGrid, ZGrid = np.meshgrid(xGrid, yGrid, zGrid, indexing='ij')
grid_points = np.stack((XGrid.ravel(), YGrid.ravel(), ZGrid.ravel()), axis=-1)

ringPos = []
ringRadius = np.ones(len(timeStamps))

# ============================ Time Loop ============================

for i in range(344, 345):  # Only 1 timestep for now
    stringtime = str(timeStampsNames[i]).zfill(4)
    print(f"Processing timestep: {stringtime}")
    filename = f'dataset2/Vortex_Ring_{stringtime}.vtp'

    try:
        X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance(filename)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        continue

    print(Radius[1][0])

    ringRadius[i], ringPos0 = rd.getRingPosRadius(X, Y, Z, Wx, Wy, Wz)
    ringPos.append(np.array(ringPos0))
    print(ringPos[-1])

    particle_pos = np.stack((X, Y, Z), axis=-1)
    Gamma = np.stack((Wx, Wy, Wz), axis=-1)

    omega_grid, omega_grid_analytical = compute_vorticity_field(
        grid_points, particle_pos, Gamma, Radius[i][0], XGrid, YGrid, ZGrid, ringPos[-1],timeStamps[i]
    )

    # Reshape to 3D grids
    omega_x = omega_grid[:, 0].reshape(XGrid.shape)
    omega_y = omega_grid[:, 1].reshape(YGrid.shape)
    omega_z = omega_grid[:, 2].reshape(ZGrid.shape)

    omega_x_analytical = omega_grid_analytical[:, 0].reshape(XGrid.shape)
    omega_y_analytical = omega_grid_analytical[:, 1].reshape(YGrid.shape)
    omega_z_analytical = omega_grid_analytical[:, 2].reshape(ZGrid.shape)

    # ============================ Plotting Cross-Section ============================

    ring_y_pos = ringPos[-1][1]
    y_plane_index = np.argmin(np.abs(yGrid - ring_y_pos))
    omega_y_slice = omega_y[:, y_plane_index, :]

    plt.figure(figsize=(8, 6))
    X, Z = np.meshgrid(xGrid, zGrid, indexing='ij')
    plt.contourf(X, Z, omega_y_slice, levels=50, cmap='viridis')
    plt.colorbar(label='ω_y [1/s]')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.title(f'Vorticity ω_y in XZ-plane at y = {ring_y_pos:.2f}')
    plt.axis('equal')
    plt.tight_layout()

    # ============================ Centerline Vorticity Plot ============================

    y_line_index = np.argmin(np.abs(yGrid - ring_y_pos))
    omega_y_line = omega_y[:, y_line_index, :]
    expected_omega_y_line = omega_y_analytical[:, y_line_index, :]

    x_line_index = np.argmin(np.abs(xGrid - ringPos[-1][0]))
    omega_y_centerline = omega_y_line[x_line_index, :]
    expected_omega_y_centerline = expected_omega_y_line[x_line_index, :]

    print(np.max(omega_y_centerline))

    plt.figure(figsize=(4, 8))
    plt.plot(omega_y_centerline, zGrid, label=r'$\omega_y$ (computed)', linewidth=3)
    plt.plot(expected_omega_y_centerline, zGrid, '--', label='Analytical', linewidth=2)
    plt.xlabel(r'$\omega_y$')
    plt.ylabel('z')
    plt.title('Centerline Vorticity ω_y')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
