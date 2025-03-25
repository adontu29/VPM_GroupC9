import numpy as np
import matplotlib.pyplot as plt

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

