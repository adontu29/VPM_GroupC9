import numpy as np
import matplotlib.pyplot as plt
import ReadData as rd
import matplotlib.animation as animation

# Define the dataset time range
timeStamps = np.arange(25, 1575, 25)

# Initialize the figure and 3D axis
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set consistent limits
ax.set_xlim(-2, 4)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)

# Initialize scatter plot (empty)
scatter = ax.scatter([], [], [], c=[], cmap="jet", marker='o')


# Function to update the animation
def update(frame):
    # Format filename properly with zero-padding
    stringtime = str(timeStamps[frame])
    zeros = ['0000', '000', '00', '0', '']  # Adjust zero-padding
    filename = f'dataset/Vortex_Ring_DNS_Re7500_{zeros[len(stringtime)]}{stringtime}.vtp'

    # Read the vortex ring data
    X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance(filename)

    # Compute updated strength magnitude
    Strength_magnitude = np.sqrt(Wx ** 2 + Wy ** 2 + Wz ** 2)

    # Update scatter plot
    scatter._offsets3d = (X, Y, Z)  # Set new positions
    scatter.set_array(Strength_magnitude)  # Update color values

    return scatter,


# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(timeStamps), interval=1, blit=False)

# Show animation
plt.show()
