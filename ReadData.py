import vtk
import numpy as np
import matplotlib.pyplot as plt

ring_center0 = np.array([0.0, 0.0, 0.0])  # m, center of the vortex ring
ring_radius0 = 1.0  # m, radius of the vortex ring
ring_strength0 = 1.0  # m²/s, vortex strength
ring_thickness0 = 0.2 * ring_radius0  # m, thickness of the vortex ring

### Particle Distribution Setup

Re = 7500  # Reynolds number

particle_distance0 = 0.25 * ring_thickness0  # m
particle_radius0 = 0.8 * particle_distance0 ** 0.5  # m
particle_viscosity0 = ring_strength0 / Re  # m²/s, kinematic viscosity
time_step_size0 = 5 * particle_distance0 ** 2 / ring_strength0  # s
n_time_steps0 = int(20 * ring_radius0 ** 2 / ring_strength0 / time_step_size0)


import vtk

# Read the VTP file
reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName("dataset/Vortex_Ring_DNS_Re7500_0075.vtp")
reader.Update()
polydata = reader.GetOutput()

# Extract points
num_points = polydata.GetNumberOfPoints()
points = np.array([polydata.GetPoint(i) for i in range(num_points)])

# Extract point data
point_data = {}
for i in range(polydata.GetPointData().GetNumberOfArrays()):
    name = polydata.GetPointData().GetArrayName(i)
    data = np.array([polydata.GetPointData().GetArray(i).GetTuple(j) for j in range(num_points)])
    point_data[name] = data

X = points[:, 0]
Y = points[:, 1]
Z = points[:, 2]

U = point_data['Velocity'][:,0]
V = point_data['Velocity'][:,1]
W = point_data['Velocity'][:,2]

Wx = point_data['Strength'][:,0]
Wy = point_data['Strength'][:,1]
Wz = point_data['Strength'][:,2]

Radius = point_data['Radius']
Group_ID = point_data['Group_ID']
Viscosity = point_data['Viscosity']
Viscosity_t = point_data['Viscosity_t']

# Print extracted data
print(f"Points array shape: {points.shape}")
print("Available point data fields:", list(point_data.keys()))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(points[:,0], points[:,1], points[:,2], marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()




