import numpy as np
import math as m
import matplotlib.pyplot as plt
import ReadData as rd
import graphing as grph

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
    Strength[i] = rd.getStrength(Wx,Wy,Wz)
    # Impulse[i] = rd.getImpulse()
    KineticEnergy[i] = rd.getKineticEnergy(X,Y,Z,Wx,Wy,Wz,Radius)
    # Helicity[i] = rd.getHelicity(X,Y,Z,Wx,Wy,Wz,Radius)
    # Enstrophy[i] = rd.getEnstrophy()

def plotytyplotplot(timeStamps, Velocity, Strength, Impulse, KineticEnergy, Helicity, Enstrophy):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))  # 15x8 inch figure

    # Flatten axs array for easier indexing

    # Plotting
    axs[0].plot(timeStamps, Velocity)
    axs[0].set_title('Velocity')

    axs[1].plot(timeStamps[1:], Strength[1:])
    axs[1].set_title('Strength')

    axs[2].plot(timeStamps, Impulse)
    axs[2].set_title('Impulse')

    axs[3].plot(timeStamps[1:], KineticEnergy[1:])
    axs[3].set_title('Kinetic Energy')

    axs[4].plot(timeStamps[1:], Helicity[1:])
    axs[4].set_title('Helicity')

    axs[5].plot(timeStamps, Enstrophy)
    axs[5].set_title('Enstrophy')

    # Adjust layout
    plt.ylim(-1, 1)
    plt.show()
        
dataset = 750

# === DATASET Re = 7500 ===

if dataset == 7500:

# === Time Steps/init ===

    ring_radius     = 1.0               # m, radius of the vortex ring
    ring_strength   = 1.0               # m²/s, vortex strength
    ring_thickness  = 0.2*ring_radius   # m, thickness of the vortex ring
    particle_distance  = 0.25*ring_thickness
    timeStep = 5 * particle_distance**2/ring_strength  # s
    timeStampsNames = np.arange(0,1575,25)
    timeStamps = timeStampsNames * timeStep
    Velocity, Strength, Impulse, KineticEnergy, Helicity, Enstrophy = getArrays(timeStampsNames)

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
        
        

        updateDiagnostics(i, X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t)

    # === Plotting ===

    timeStamps = timeStampsNames * timeStep
    

# === DATASET Re = 750 ===

elif dataset == 750:

# === Time Steps/init ===

    timeStep = 5.808E-03 * 25 #s
    timeStampsNames = np.arange(0,8600,25)
    Velocity, Strength, Impulse, KineticEnergy, Helicity, Enstrophy = getArrays(timeStampsNames)

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

        updateDiagnostics(i, X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t)

    # === Plotting ===

    timeStamps = timeStampsNames * timeStep
    
    with open ("kineticEnergyResults.txt","w") as file:
        for i  in range(len(KineticEnergy)):
            file.write(str(i) + " " + str(KineticEnergy[i]) + "\n")

    with open ("strengthResults.txt","w") as file:
        for i  in range(len(Strength)):
            file.write(str(i) + " " + str(Strength[i]) + "\n")
    
    