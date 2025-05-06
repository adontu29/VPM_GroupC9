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

def plotytyplotplot(timeStamps, Velocity, Strength, Impulse, KineticEnergy, Helicity, Enstrophy):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))  # 15x8 inch figure

    # Flatten axs array for easier indexing
    axs = axs.flatten()

    # Plotting
    axs[0].plot(timeStamps, Velocity)
    axs[0].set_title('Velocity')

    axs[1].plot(timeStamps, Strength)
    axs[1].set_title('Strength')

    axs[2].plot(timeStamps, Impulse)
    axs[2].set_title('Impulse')

    axs[3].plot(timeStamps, KineticEnergy)
    axs[3].set_title('Kinetic Energy')

    axs[4].plot(timeStamps, Helicity)
    axs[4].set_title('Helicity')

    axs[5].plot(timeStamps, Enstrophy)
    axs[5].set_title('Enstrophy')

    # Adjust layout
    plt.tight_layout()
    plt.show()
        
dataset = 7500

# === DATASET Re = 7500 ===

if dataset == 7500:

# === Time Steps ===

    ring_radius     = 1.0               # m, radius of the vortex ring
    ring_strength   = 1.0               # m²/s, vortex strength
    ring_thickness  = 0.2*ring_radius   # m, thickness of the vortex ring
    particle_distance  = 0.25*ring_thickness
    timeStep = 5 * particle_distance**2/ring_strength  # s
    timeStampsNames = np.arange(0,1575,25)
    timeStamps = timeStampsNames * timeStep

# === Iteration per file ===

    for i in range(len(timeStampsNames)):
        print("File: " + str(timeStampsNames[i]))

        # === Filenames + Data Extraction === 
    
        stringtime = str(timeStampsNames[i]).zfill(4)
        # Debugged: Ensure correct file path format
        filename = f'dataset/Vortex_Ring_DNS_Re7500_{stringtime}.vtp'

        try:
            X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance(filename)
        except FileNotFoundError:
            print(f"Error: File {filename} not found.")
            continue

        # === Diagnostics Updated From ReadData.py ===
        
        Velocity, Strength, Impulse, KineticEnergy, Helicity, Enstrophy = getArrays(timeStampsNames)

        updateDiagnostics(i, X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t)

    # === Plotting ===

    timeStamps = timeStampsNames * timeStep
    plotytyplotplot(timeStamps, Velocity, Strength, Impulse, KineticEnergy, Helicity, Enstrophy)

# === DATASET Re = 750 ===

elif dataset == 750:

# === Time Steps ===

    timeStep = 5.808E-03 * 25 #s
    timeStampsNames = np.arange(0,8600,25)

# === Iteration per file ===

    for i in range(len(timeStampsNames)):
        print("File: " + str(timeStampsNames[i]))
        # === Filenames + Data Extraction === 
                
        stringtime = str(timeStampsNames[i]).zfill(4)
        # Debugged: Ensure correct file path format
        filename = f'dataset2/Vortex_Ring_{stringtime}.vtp'

        try:
            X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance(filename)
        except FileNotFoundError:
            print(f"Error: File {filename} not found.")
            continue
        
        # === Diagnostics Updated From ReadData.py ===

        Velocity, Strength, Impulse, KineticEnergy, Helicity, Enstrophy = getArrays(timeStampsNames)

        updateDiagnostics(i, X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t)

    # === Plotting ===

    timeStamps = timeStampsNames * timeStep
    plotytyplotplot(timeStamps, Velocity, Strength, Impulse, KineticEnergy, Helicity, Enstrophy)


for i in range(len(timeStampsNames)):
    print("peepee")



