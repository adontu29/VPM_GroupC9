import numpy as np
import math as m
import matplotlib.pyplot as plt
import ReadData as rd

def zeta(rho):
    return 15 / (8 * np.pi * (rho**2 + 1)**(7/2))

def q(rho):
    return rho**3 * (rho**2 + 2.5) / ((rho**2 + 1)**(5/2)) / (4 * np.pi)

def reg_zeta(rho, radius):
    rho = rho / radius
    return 15 / (8 * np.pi * (rho**2 + 1)**(7/2)) / radius**3

def reg_q(rho, radius):
    rho = rho / radius
    return (rho**3 * (rho**2 + 2.5)) / ((rho**2 + 1)**(5/2)) / (4 * np.pi)

def compute_vorticity_field(grid_points, particle_pos, Gamma, particle_radius):
    """
    Computes the vorticity field at each grid point due to vortex particles.
    """
    omega_grid = np.zeros_like(grid_points)

    for p in range(particle_pos.shape[0]):
        print(p, particle_pos.shape[0])
        r = grid_points - particle_pos[p]          # (M, 3)
        r_mag = np.linalg.norm(r, axis=1)          # (M,)
        r_mag[r_mag == 0] = 1e-12                   # avoid division by zero

        r_squared = r_mag**2
        r_cubed = r_mag**3

        zeta_val = reg_zeta(r_mag, particle_radius)  # (M,)
        q_val = reg_q(r_mag, particle_radius)        # (M,)

        dot_rg = np.einsum('ij,j->i', r, Gamma[p])   # r · Γ  → (M,)

        # First term
        term1 = (zeta_val - q_val / r_cubed)[:, None] * Gamma[p]  # (M, 3)

        # Second term
        scalar_coeff = (3 * q_val / r_cubed - zeta_val) * (dot_rg / r_squared)  # (M,)
        term2 = scalar_coeff[:, None] * r                                       # (M, 3)

        omega_grid += term1 + term2

    return omega_grid


# ========== Simulation Setup ==========

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

# Grid setup
xGrid = np.arange(-1.5, 1.5, 0.1)
yGrid = np.arange(-0.3, 4.3, 0.1)
zGrid = np.arange(-1.5, 1.5, 0.1)
XGrid, YGrid, ZGrid = np.meshgrid(xGrid, yGrid, zGrid, indexing='ij')
grid_points = np.stack((XGrid.ravel(), YGrid.ravel(), ZGrid.ravel()), axis=-1)

ringPos = []
ringRadius = np.ones(len(timeStamps))


# ========== Time Loop ==========
for i in range(0,1):
    stringtime = str(timeStampsNames[i]).zfill(4)
    print(stringtime)
    filename = f'dataset2/Vortex_Ring_{stringtime}.vtp'

    try:
        X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance(filename)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        continue
    ringRadius[i], ringPos0 = rd.getRingPosRadius(X, Y, Z, Wx, Wy, Wz)
    ringPos.append(np.array(ringPos0))

    particle_pos = np.stack((X, Y, Z), axis=-1)
    Gamma = np.stack((Wx, Wy, Wz), axis=-1)

    omega_grid = compute_vorticity_field(grid_points, particle_pos, Gamma, particle_radius)

    # Reshape to grid
    omega_x = omega_grid[:, 0].reshape(XGrid.shape)
    omega_y = omega_grid[:, 1].reshape(YGrid.shape)
    omega_z = omega_grid[:, 2].reshape(ZGrid.shape)

    # === Plot vorticity in cross-section perpendicular to y-axis at the vortex center ===

    # Find index in the grid that is closest to the vortex ring center's y-coordinate
    ring_y_pos = ringPos[-1][1]
    y_plane_index = np.argmin(np.abs(yGrid - ring_y_pos))

    # Extract the XZ slice at y = ring center
    omega_x_slice = omega_x[:, y_plane_index, :]  # Shape: (len(xGrid), len(zGrid))
    omega_y_slice = omega_y[:, y_plane_index, :]
    omega_z_slice = omega_z[:, y_plane_index, :]

    # Compute vorticity magnitude in the slice
    omega_mag_slice = np.sqrt(omega_x_slice**2 + omega_y_slice**2 + omega_z_slice**2)

    # Plotting
    plt.figure(figsize=(8, 6))
    X, Z = np.meshgrid(xGrid, zGrid, indexing='ij')
    plt.contourf(X, Z, omega_y_slice, levels=50, cmap='viridis')
    plt.colorbar(label='|ω| [1/s]')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.title(f'Vorticity Magnitude in XZ-plane at y = {ring_y_pos:.2f}')
    plt.axis('equal')
    plt.tight_layout()


    # === Extract centerline vorticity along y (x = ring_x_pos) in XY-plane at z = ring_z_pos ===

    # === Extract centerline vorticity along z (y = ring_y_pos) in XZ-plane at y = ring_y_pos ===

    ring_y_pos = ringPos[-1][1]
    y_line_index = np.argmin(np.abs(yGrid - ring_y_pos))

    # Extract ω_y along y = center line in the slice
    omega_y_line = omega_y[:, y_line_index, :]

    # Choose a line at the center of that slice (e.g. at x = ring_x_pos)
    x_line_index = np.argmin(np.abs(xGrid - ringPos[-1][0]))
    omega_y_centerline = omega_y_line[x_line_index, :]  # (along z)

    # Optional: expected profile
    expected_omega_y = np.zeros_like(zGrid)  # placeholder

    # Plotting
    plt.figure(figsize=(4, 8))
    plt.plot(omega_y_centerline, zGrid, label=r'$\omega_y$ along centerline', linewidth=3)
    plt.plot(expected_omega_y, zGrid, '--', label='Expected profile', linewidth=2)
    plt.xlabel(r'$\omega_y$')
    plt.ylabel('z')
    plt.title('Centerline Vorticity (ω_y)')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()



