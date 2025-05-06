import numpy as np
import math as m
import matplotlib.pyplot as plt
import ReadData as rd

def getArrays(timeStampsNames):
    Velocity = np.zeros((len(timeStampsNames)))
    Strength = np.zeros((len(timeStampsNames)))
    Impulse = np.zeros((len(timeStampsNames)))
    KineticEnergy = np.zeros((len(timeStampsNames)))
    Helicity = np.zeros((len(timeStampsNames)))
    Enstrophy = np.zeros((len(timeStampsNames)))
    return Velocity, Strength, Impulse, KineticEnergy, Helicity, Enstrophy

def updateDiagnostics(i,X,Y,Z,U,V,W,Wx,Wy,Wz,Radius,Group_ID,Viscosity,Viscosity_t):
    # Velocity[i] = 0
    # Strength[i] = rd.getStrength()
    # Impulse[i] = rd.getImpulse()
    KineticEnergy[i] = rd.getKineticEnergy(X,Y,Z,Wx,Wy,Wz,Radius)
    # Helicity[i] = rd.getHelicity()
    # Enstrophy[i] = rd.getEnstrophy()
        
dataset = 7500

if dataset == 7500:
    timeStampsNames = np.arange(0,1575,25)

    for i in range(len(timeStampsNames)):
        stringtime = str(timeStampsNames[i]).zfill(4)
        # Debugged: Ensure correct file path format
        filename = f'dataset/Vortex_Ring_DNS_Re7500_{stringtime}.vtp'

        try:
            X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance(filename)
        except FileNotFoundError:
            print(f"Error: File {filename} not found.")
            continue
        
        Velocity, Strength, Impulse, KineticEnergy, Helicity, Enstrophy = getArrays(timeStampsNames)

        updateDiagnostics(i, X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t)

elif dataset == 750:

    timeStampsNames = np.arange(0,8600,25)

    for i in range(len(timeStampsNames)):
        stringtime = str(timeStampsNames[i]).zfill(4)
        # Debugged: Ensure correct file path format
        filename = f'dataset2/Vortex_Ring_{stringtime}.vtp'

        try:
            X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance(filename)
        except FileNotFoundError:
            print(f"Error: File {filename} not found.")
            continue

        Velocity, Strength, Impulse, KineticEnergy, Helicity, Enstrophy = getArrays(timeStampsNames)

        updateDiagnostics(i, X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t)


for i in range(len(timeStampsNames)):
    print("peepee")



