import numpy as np
import matplotlib.pyplot as plt
from Circulation_Impulse import compute_linear_impulse, compute_angular_impulse

# ==== Load or generate your time-dependent vortex data ====
# These should be lists of arrays, one per time step
# Replace with your actual data loading if you have it from files

timesteps = 10
X_t = [np.random.rand(100) for _ in range(timesteps)]
Y_t = [np.random.rand(100) for _ in range(timesteps)]
Z_t = [np.random.rand(100) for _ in range(timesteps)]
Wx_t = [np.random.rand(100) for _ in range(timesteps)]
Wy_t = [np.random.rand(100) for _ in range(timesteps)]
Wz_t = [np.random.rand(100) for _ in range(timesteps)]

# ==== Compute impulses over time ====

linear_impulse_over_time = []
angular_impulse_over_time = []

for i in range(timesteps):
    L = compute_linear_impulse(X_t[i], Y_t[i], Z_t[i], Wx_t[i], Wy_t[i], Wz_t[i])
    A = compute_angular_impulse(X_t[i], Y_t[i], Z_t[i], Wx_t[i], Wy_t[i], Wz_t[i])
    linear_impulse_over_time.append(L)
    angular_impulse_over_time.append(A)

linear_impulse_over_time = np.array(linear_impulse_over_time)
angular_impulse_over_time = np.array(angular_impulse_over_time)

# ==== Time array (adjust if needed) ====
time_array = np.arange(timesteps) * 0.025  # in seconds

# ==== Plotting ====

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# --- Linear Impulse Plot ---
axs[0].plot(time_array, linear_impulse_over_time[:, 0], label='Ix')
axs[0].plot(time_array, linear_impulse_over_time[:, 1], label='Iy')
axs[0].plot(time_array, linear_impulse_over_time[:, 2], label='Iz')
axs[0].set_ylabel('Linear Impulse')
axs[0].set_title('Linear Impulse over Time')
axs[0].legend()
axs[0].grid(True)

# --- Angular Impulse Plot ---
axs[1].plot(time_array, angular_impulse_over_time[:, 0], label='Jx')
axs[1].plot(time_array, angular_impulse_over_time[:, 1], label='Jy')
axs[1].plot(time_array, angular_impulse_over_time[:, 2], label='Jz')
axs[1].set_ylabel('Angular Impulse')
axs[1].set_xlabel('Time [s]')
axs[1].set_title('Angular Impulse over Time')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
