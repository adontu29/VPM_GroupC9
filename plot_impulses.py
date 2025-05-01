import numpy as np
import matplotlib.pyplot as plt
import ReadData2 as rd
from vtriClass import VortexRingInstance
from Circulation_Impulse import compute_linear_impulse, compute_angular_impulse

# Configuration
DATA_PATH = "dataset"
FILENAME_TEMPLATE = "Vortex_Ring_DNS_Re7500_{:04d}.vtp"
TIMESTAMPS = np.arange(25, 1575, 25)  # time values, step of 25
DT = 0.005  # Optional, used for time axis scaling if needed

# Storage for results
linear_impulses = []
angular_impulses = []

# Process each timestep
for stamp in TIMESTAMPS:
    print(f"Processing timestep {stamp}")
    filepath = f"{DATA_PATH}/{FILENAME_TEMPLATE.format(stamp)}"

    # Load data for this timestep
    x, y, z, u, v, w, wx, wy, wz, radius, group_id, mu, mu_t = rd.readVortexRingInstance(filepath)
    ring = VortexRingInstance(x, y, z, u, v, w, wx, wy, wz, radius, group_id, mu, mu_t)

    # Compute impulse
    L = compute_linear_impulse(ring.x, ring.y, ring.z, ring.wx, ring.wy, ring.wz)
    A = compute_angular_impulse(ring.x, ring.y, ring.z, ring.wx, ring.wy, ring.wz)

    linear_impulses.append(L)
    angular_impulses.append(A)

# Convert to NumPy arrays
linear_impulses = np.array(linear_impulses)
angular_impulses = np.array(angular_impulses)

# Convert TIMESTAMPS to seconds (optional)
times = TIMESTAMPS * DT

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Linear Impulse Plot
axs[0].plot(times, linear_impulses[:, 0], label="Ix")
axs[0].plot(times, linear_impulses[:, 1], label="Iy")
axs[0].plot(times, linear_impulses[:, 2], label="Iz")
axs[0].set_ylabel("Linear Impulse")
axs[0].set_title("Linear Impulse Over Time")
axs[0].legend()
axs[0].grid(True)

# Angular Impulse Plot
axs[1].plot(times, angular_impulses[:, 0], label="Jx")
axs[1].plot(times, angular_impulses[:, 1], label="Jy")
axs[1].plot(times, angular_impulses[:, 2], label="Jz")
axs[1].set_ylabel("Angular Impulse")
axs[1].set_xlabel("Time [s]")
axs[1].set_title("Angular Impulse Over Time")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
