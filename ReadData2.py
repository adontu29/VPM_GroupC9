import vtk
import numpy as np
import matplotlib.pyplot as plt

ring_center0 = np.array([0.0, 0.0, 0.0])  # m, center of the vortex ring
ring_radius0 = 1.0  # m, radius of the vortex ring
ring_strength0 = 1.0  # m²/s, vortex strength
ring_thickness0 = 0.2 * ring_radius0  # m, thickness of the vortex ring

# Particle Distribution Setup

Re = 7500  # Reynolds number

particle_distance0 = 0.25 * ring_thickness0  # m
particle_radius0 = 0.8 * particle_distance0 ** 0.5  # m
particle_viscosity0 = ring_strength0 / Re  # m²/s, kinematic viscosity
time_step_size0 = 5 * particle_distance0 ** 2 / ring_strength0  # s
n_time_steps0 = int(20 * ring_radius0 ** 2 / ring_strength0 / time_step_size0)


# Read the VTP file

def readVortexRingInstance(filename):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
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

    U = point_data['Velocity'][:, 0]
    V = point_data['Velocity'][:, 1]
    W = point_data['Velocity'][:, 2]

    Wx = point_data['Strength'][:, 0]
    Wy = point_data['Strength'][:, 1]
    Wz = point_data['Strength'][:, 2]

    Radius = point_data['Radius']
    Group_ID = point_data['Group_ID']
    Viscosity = point_data['Viscosity']
    Viscosity_t = point_data['Viscosity_t']

    return [X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t]


def getRingPosRadius(X, Y, Z, Wx, Wy, Wz):
    Strength_magnitude = np.sqrt(np.square(Wx) + np.square(Wy) + np.square(Wz))
    PositionVector = np.sqrt(Y ** 2 + Z ** 2)
    maxStrength = np.max(Strength_magnitude)
    Threshold = 0
    X = X[Strength_magnitude > maxStrength * Threshold]
    Y = Y[Strength_magnitude > maxStrength * Threshold]
    Z = Z[Strength_magnitude > maxStrength * Threshold]
    PositionVector = PositionVector[Strength_magnitude > maxStrength * Threshold]
    Strength_magnitude = Strength_magnitude[Strength_magnitude > maxStrength * Threshold]

    Strength_total = sum(Strength_magnitude)
    X_avg = 0
    Y_avg = 0
    Z_avg = 0
    Radius_avg = 0
    for i in range(len(Strength_magnitude)):
        weight = Strength_magnitude[i]
        X_avg += X[i] * weight
        Y_avg += Y[i] * weight
        Z_avg += Z[i] * weight
        Radius_avg += PositionVector[i] * weight

    X_avg /= Strength_total
    Y_avg /= Strength_total
    Z_avg /= Strength_total
    Radius_avg /= Strength_total
    VortexRingPosition = [X_avg, Y_avg, Z_avg]

    return Radius_avg, VortexRingPosition


def compute_ring_trajectory(filenames, particle_radius, core_radius, dt):
    positions = []

    for file in filenames:
        X, Y, Z, U, V, W, Wx, Wy, Wz, *_ = readVortexRingInstance(file)

        # Get ring position
        _, pos = getRingPosRadius(X, Y, Z, Wx, Wy, Wz)
        positions.append(pos)

    positions = np.array(positions)

    velocities = np.gradient(positions, dt, axis=0)  # shape (T, 3)

    accelerations = np.gradient(velocities, dt, axis=0)  # shape (T, 3)

    return positions, velocities, accelerations