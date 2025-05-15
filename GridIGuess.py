import numpy as np
import ReadData as rd

# Define the ranges for x, y, and z axes
xGrid = np.arange(-1.5, 1.5, 0.1)
yGrid = np.arange(-0.3, 4.3, 0.1)
zGrid = np.arange(-1.5, 1.5, 0.1)

# Create the 3D grid
XGrid, YGrid, ZGrid = np.meshgrid(xGrid, yGrid, zGrid, indexing='ij')  # 'ij' gives matrix indexing

# Now X, Y, Z are 3D arrays of shape (5, 5, 5)
print("X shape:", XGrid.shape)
timeStamps = np.arange(0,1575,25)
Velocity = np.ones(len(timeStamps))

for i in range(len(timeStamps)):
    print(timeStamps[i])
    stringtime = str(timeStamps[i]).zfill(4)
    # Debugged: Ensure correct file path format
    filename = f'dataset/Vortex_Ring_DNS_Re7500_{stringtime}.vtp'

    try:
        X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance(filename)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        continue

    # Debugged: Ensure `gamma[i]` is properly computed
    # Stack the grid into shape (N_grid, 3)
    grid_shape = XGrid.shape
    grid_points = np.stack((XGrid.ravel(), YGrid.ravel(), ZGrid.ravel()), axis=-1)

    # Stack particle data
    particle_pos = np.stack((X, Y, Z), axis=-1)  # shape: (N_particles, 3)
    Gamma = np.stack((Wx, Wy, Wz), axis=-1)  # shape: (N_particles, 3), already circulation

    # Initialize the velocity at each grid point
    u_grid = np.zeros_like(grid_points)

    # Loop over each vortex particle
    for p in range(particle_pos.shape[0]):
        r = grid_points - particle_pos[p]  # (N_grid, 3)
        r_mag = np.linalg.norm(r, axis=1)  # |r|, shape: (N_grid,)

        # Avoid divide-by-zero: optionally mask or set a min value
        r_mag[r_mag == 0] = 1e-12

        r_cubed = r_mag ** 3  # |r|^3
        K = -r / (4 * np.pi * r_cubed[:, np.newaxis])  # Biot–Savart kernel
        u_contrib = np.cross(K, Gamma[p])  # K × Γ
        u_grid += u_contrib

    # Reshape velocity field back to 3D grid
    Ux = u_grid[:, 0].reshape(grid_shape)
    Uy = u_grid[:, 1].reshape(grid_shape)
    Uz = u_grid[:, 2].reshape(grid_shape)

    # Assume Ux, Uy, Uz are 3D arrays of velocity on the same grid as XGrid, YGrid, ZGrid
    dx = xGrid[1] - xGrid[0]
    dy = yGrid[1] - yGrid[0]
    dz = zGrid[1] - zGrid[0]

    # Compute gradients (∂u/∂x, etc.)
    dUz_dy, dUz_dx, dUz_dz = np.gradient(Uz, dy, dx, dz)
    dUy_dz, dUy_dx, dUy_dy = np.gradient(Uy, dz, dx, dy)
    dUx_dz, dUx_dx, dUx_dy = np.gradient(Ux, dz, dx, dy)

    # Curl: ω = ∇ × u
    Wx_grid = dUz_dy - dUy_dz
    Wy_grid = dUx_dz - dUz_dx
    Wz_grid = dUy_dx - dUx_dy

    # Find index of plane closest to x = 0
    x_index = np.argmin(np.abs(xGrid))

    # Take absolute value of ω_x in x=0 plane
    Wx_plane = np.abs(Wx_grid[x_index, :, :])

    # Circulation estimate (divide by 2 to account for symmetry)
    Gamma = 0.5 * np.sum(Wx_plane) * dy * dz

    print(f"Estimated vortex ring circulation: {Gamma:.5f}")


[4.98241,4.11510,2.10322,0.86489,1.00185,1.05919,0.86324,0.73651,0.64519,0.56933,0.50394,0.44722]