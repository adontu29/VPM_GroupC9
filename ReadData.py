import math

import vtk
import numpy as np
import matplotlib.pyplot as plt
from numba import jit,njit

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

    return [X,Y,Z,U,V,W,Wx,Wy,Wz,Radius,Group_ID,Viscosity,Viscosity_t]


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

def getRingCoreRadius0(X, Y, Z, Wx, Wy, Wz , RingPos, RingRadius):
    Strength_magnitude = np.sqrt(np.square(Wx) + np.square(Wy) + np.square(Wz))
    maxStrength = np.max(Strength_magnitude)
    Threshold = 0
    X = X[Strength_magnitude > maxStrength * Threshold] - RingPos[0]
    Y = Y[Strength_magnitude > maxStrength * Threshold] - RingPos[1]
    Z = Z[Strength_magnitude > maxStrength * Threshold] - RingPos[2]
    Radius = np.sqrt(np.square(Y) + np.square(Z))
    coreRadiusZ = (np.max(np.abs(Z)) - np.min(np.abs(Z)))/2
    coreRadiusR = (np.max(Radius) - np.min(Radius))/2
    return coreRadiusR

def getRingCoreRadius(X, Y, Z, Wx, Wy, Wz , RingPos, RingRadius):
    Strength_magnitude = np.sqrt(np.square(Wx) + np.square(Wy) + np.square(Wz))
    Strength_total = sum(Strength_magnitude)
    maxStrength = np.max(Strength_magnitude)
    Threshold = 0
    X = X[Strength_magnitude > maxStrength * Threshold] - RingPos[0]
    Y = Y[Strength_magnitude > maxStrength * Threshold] - RingPos[1]
    Z = Z[Strength_magnitude > maxStrength * Threshold] - RingPos[2]
    Radius = np.sqrt(np.square(Y) + np.square(Z))
    coreRadiusZ = (np.max(np.abs(Z)) - np.min(np.abs(Z)))/2
    coreRadiusR = (np.max(Radius) - np.min(Radius))/2

    Core_radius_avg = 0
    for i in range(len(Strength_magnitude)):
        weight = Strength_magnitude[i]
        Core_radius_avg += np.abs(Radius[i]-RingRadius) * weight

    Core_radius_avg /= Strength_total
    return Core_radius_avg

def getRingStrength(X, Y, Z, Wx, Wy, Wz, RingPos,particleRadius, coreRadius):
    # vorticity_magnitude = (ring_strength/(np.pi*ring_thickness**2)) * np.exp(-radial_distance_to_core**2/ring_thickness**2)
    particleVolume = 4/3 * np.pi * particleRadius**3
    strengthMagnitude = np.sqrt(np.square(Wx) + np.square(Wy) + np.square(Wz))
    vorticityMagnitude = strengthMagnitude/particleVolume
    X = X - RingPos[0]
    Y = Y - RingPos[1]
    Z = Z - RingPos[2]
    radialDistance = np.sqrt(np.square(X) + np.square(Y) + np.square(Z))
    ringStrength = vorticityMagnitude*(np.exp(radialDistance**2/(coreRadius*2)**2))*(np.pi*(coreRadius*2)**2)

    return np.mean(ringStrength)

@jit
def getKineticEnergy(X,Y,Z,Wx,Wy,Wz,radius):
    n = len(X)
    sig = radius[0, 0]
    E = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = X[i] - X[j]
            dy = Y[i] - Y[j]
            dz = Z[i] - Z[j]
            norm_sq = dx*dx + dy*dy + dz*dz
            norm = np.sqrt(norm_sq)
            rho = norm / sig
            dot_apaq = Wx[i] * Wx[j] + Wy[i] * Wy[j] + Wz[i] * Wz[j]
            dot_diffap = Wx[i] * dx + Wy[i] * dy + Wz[i] * dz
            dot_diffaq = Wx[j] * dx + Wy[j] * dy + Wz[j] * dz

            term1 = 2 * rho / np.sqrt(rho ** 2 + 1) * dot_apaq

            raw = (dot_diffap * dot_diffaq) / norm_sq
            term2 = rho ** 3 / ((rho ** 2 + 1) ** 1.5) * (raw - dot_apaq)

            contribution = (term1 + term2) / norm
            E += contribution

            # # ap[i] and aq[j] dot product
            # dot_apaq = Wx[i]*Wx[j] + Wy[i]*Wy[j] + Wz[i]*Wz[j]
            #
            # # dot_diffap and dot_diffaq
            # dot_diffap = Wx[i]*dx + Wy[i]*dy + Wz[i]*dz
            # dot_diffaq = Wx[j]*dx + Wy[j]*dy + Wz[j]*dz
            #
            # term1 = (2*rho)/np.sqrt(rho**2 + 1) * dot_apaq
            # term2 = (rho**3)/( (rho**2 + 1)**1.5 ) * (dot_diffap * dot_diffaq) / norm_sq
            # contribution = (1.0 / norm) * (term1 + term2 - dot_apaq)
            # E += contribution

        # Optional progress update (disable in performance runs)
        if i % 250 == 0:
            print(i)  # or: print(i) if testing outside @njit

    E = E / (16 * np.pi)
    return E

@jit
def getStrength(Wx, Wy, Wz):
    print("Getting Strength ... ")
    strength = np.sum(np.sqrt(Wx ** 2 + Wy ** 2 + Wz ** 2) / (2 * np.pi)) # 1/s
    return strength