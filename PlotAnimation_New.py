import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import ReadData2 as rd

# === Configuration ===
DATA_PATH = "dataset"
FILENAME_TEMPLATE = "Vortex_Ring_DNS_Re7500_{:04d}.vtp"
TIMESTAMPS = np.arange(25, 1575, 25)  # in steps of 25

# === Visualization Setup ===
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.set_title("Vortex Ring Strength Evolution")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-2, 4)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)

# Initialize empty scatter plot
scatter = ax.scatter([], [], [], c=[], cmap='jet', marker='o', s=10)
colorbar = plt.colorbar(scatter, ax=ax, pad=0.1, label='Strength Magnitude')


# === Animation Update Function ===
def update(frame_idx):
    timestep = TIMESTAMPS[frame_idx]
    filename = f"{DATA_PATH}/{FILENAME_TEMPLATE.format(timestep)}"

    try:
        # Read vortex ring data
        X, Y, Z, U, V, W, Wx, Wy, Wz, _, _, _, _ = rd.readVortexRingInstance(filename)

        # Compute vorticity strength magnitude
        strength = np.sqrt(Wx**2 + Wy**2 + Wz**2)

        # Update scatter plot
        scatter._offsets3d = (X, Y, Z)
        scatter.set_array(strength)
        scatter.set_clim(strength.min(), strength.max())  # Optional: adjust colormap scale dynamically

        ax.set_title(f"Time = {timestep}")
    except Exception as e:
        print(f"Error reading {filename}: {e}")

    return scatter,


# === Run Animation ===
ani = animation.FuncAnimation(fig, update, frames=len(TIMESTAMPS), interval=50, blit=False)

plt.tight_layout()
plt.show()
