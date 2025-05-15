import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import ReadData2 as rd

# === Configuration ===
DATA_PATH = "dataset"
FILENAME_TEMPLATE = "Vortex_Ring_DNS_Re7500_{:04d}.vtp"
TIMESTAMPS = np.arange(25, 1575, 25)  # in steps of 25
print(len(TIMESTAMPS))
colors = ['#1B2B42', '#2D5D7A', '#4F9A8D', '#76C28A']


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



# Input data into graph

# timestep = TIMESTAMPS[1]
# filename = f"{DATA_PATH}/{FILENAME_TEMPLATE.format(timestep)}"
# Xs = np.zeros((len(TIMESTAMPS), 8))
# Ys = np.zeros((len(TIMESTAMPS), 8))
# Zs = np.zeros((len(TIMESTAMPS), 8))
# Read vortex ring data
for i in range(4):
    timestep = TIMESTAMPS[15*i]
    filename = f"{DATA_PATH}/{FILENAME_TEMPLATE.format(timestep)}"
    X, Y, Z, U, V, W, Wx, Wy, Wz, _, _, _, _ = rd.readVortexRingInstance(filename)

    # Update scatter plot
    scatter = ax.scatter([X], [Y], [Z], c=colors[i], cmap='jet', marker='o', s=10)
    #colorbar = plt.colorbar(scatter, ax=ax, pad=0.1, label='Strength Magnitude')

plt.show()