import numpy as np
import matplotlib.pyplot as plt
from Circulation_Impulse import compute_linear_impulse, compute_angular_impulse

def load_vortex_data(timesteps: int, N_particles: int):
    """
    Placeholder function to generate synthetic vortex particle data.
    Replace this with actual data loading for physical simulation.
    """
    X_t = [np.random.rand(N_particles) for _ in range(timesteps)]
    Y_t = [np.random.rand(N_particles) for _ in range(timesteps)]
    Z_t = [np.random.rand(N_particles) for _ in range(timesteps)]
    Wx_t = [np.random.rand(N_particles) for _ in range(timesteps)]
    Wy_t = [np.random.rand(N_particles) for _ in range(timesteps)]
    Wz_t = [np.random.rand(N_particles) for _ in range(timesteps)]
    return X_t, Y_t, Z_t, Wx_t, Wy_t, Wz_t

def validate_data(*arrays):
    """
    Ensure all time-dependent lists have the same length.
    """
    length = len(arrays[0])
    if not all(len(arr) == length for arr in arrays):
        raise ValueError("Time series arrays are not the same length.")

def compute_impulses_over_time(X_t, Y_t, Z_t, Wx_t, Wy_t, Wz_t):
    """
    Compute linear and angular impulses over time.
    """
    linear_impulse = []
    angular_impulse = []
    for x, y, z, wx, wy, wz in zip(X_t, Y_t, Z_t, Wx_t, Wy_t, Wz_t):
        L = compute_linear_impulse(x, y, z, wx, wy, wz)
        A = compute_angular_impulse(x, y, z, wx, wy, wz)
        linear_impulse.append(L)
        angular_impulse.append(A)
    return np.array(linear_impulse), np.array(angular_impulse)

def plot_impulses(time_array, linear_impulse, angular_impulse):
    """
    Plot linear and angular impulse over time.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Linear Impulse
    axs[0].plot(time_array, linear_impulse[:, 0], label='Ix')
    axs[0].plot(time_array, linear_impulse[:, 1], label='Iy')
    axs[0].plot(time_array, linear_impulse[:, 2], label='Iz')
    axs[0].set_ylabel('Linear Impulse')
    axs[0].set_title('Linear Impulse over Time')
    axs[0].legend()
    axs[0].grid(True)

    # Angular Impulse
    axs[1].plot(time_array, angular_impulse[:, 0], label='Jx')
    axs[1].plot(time_array, angular_impulse[:, 1], label='Jy')
    axs[1].plot(time_array, angular_impulse[:, 2], label='Jz')
    axs[1].set_ylabel('Angular Impulse')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_title('Angular Impulse over Time')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def main():
    # === Simulation parameters ===
    timesteps = 50
    N_particles = 100
    dt = 0.005  # seconds

    # === Load or generate vortex data ===
    X_t, Y_t, Z_t, Wx_t, Wy_t, Wz_t = load_vortex_data(timesteps, N_particles)

    # === Validate data consistency ===
    validate_data(X_t, Y_t, Z_t, Wx_t, Wy_t, Wz_t)

    # === Compute impulses ===
    linear_impulse, angular_impulse = compute_impulses_over_time(X_t, Y_t, Z_t, Wx_t, Wy_t, Wz_t)

    # === Generate time array ===
    time_array = np.arange(timesteps) * dt

    # === Plot impulses ===
    plot_impulses(time_array, linear_impulse, angular_impulse)

if __name__ == "__main__":
    main()
