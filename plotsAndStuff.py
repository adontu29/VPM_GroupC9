import graphing
import numpy as np
from matplotlib import pyplot as plt

# =========================
# Load kinetic energy data
# =========================
with open("kineticEnergyResults.txt", "r") as f:
    content = f.readlines()
    kineticEnergy = []
    for line in content:
        index = line.find(" ")
        value = line[index + 1:].strip() if index != -1 else line.strip()
        kineticEnergy.append(float(value))

# =======================
# Load strength data
# =======================
with open("strengthResults.txt", "r") as f:
    content = f.readlines()
    strength = []
    for line in content:
        index = line.find(" ")
        value = line[index + 1:].strip() if index != -1 else line.strip()
        strength.append(float(value))

# =========================
# Vortex ring and particle setup
# =========================
ring_center     = np.array([0.0, 0.0, 0.0])   # m
ring_radius     = 1.0                         # m
ring_strength   = 1.0                         # m²/s
ring_thickness  = 0.2 * ring_radius           # m

Re = 750                                   # Reynolds number
particle_distance  = 0.22*ring_thickness    # m
particle_radius    = 0.8*particle_distance**0.5  # m
particle_viscosity = ring_strength/Re       # m²/s, kinematic viscosity
time_step_size     = 25 * 3 * particle_distance**2/ring_strength  # s
n_time_steps       = int( 100*ring_radius**2 / ring_strength / time_step_size)
max_timesteps = 8600
no_timesteps = 8600/25+1


timeStampsNames = np.arange(0,max_timesteps+1,25)
timeStamps = np.arange(0, no_timesteps*time_step_size, time_step_size)

# ======================
# Time and velocity calculation
# ======================
saffmanEnergy = np.zeros(len(timeStamps))
for i in range(len(timeStamps)):
    eps = 1e-8
    C = -2.05
    ring_thickness_t = np.sqrt(4 * particle_viscosity * timeStamps[i] + ring_thickness**2)
    term_a = 0.5 * ring_strength**2 * ring_radius
    term_b = np.log(8 * ring_radius / ring_thickness_t) + C
    saffmanEnergy[i] = term_a * term_b
    
# ======================
# Plotting
# ======================
# If you want to compare both energies:
graphing.getDoubleGraph(timeStamps[::5], timeStamps[::5], kineticEnergy[::5], saffmanEnergy[::5],
                    "Kinetic Energy", "Saffman Model Kinetic Energy", 2, "r", "b", "Time (s)", "Kinetic Energy (J)")
plt.show()
print(timeStamps)
# If just plotting kinetic energy:
# graphing.getGraph(timeStamps, kineticEnergy, "Kinetic Energy", 0.35)
# graphing.getGraph(timeStamps[::5], strength[::5], "Strength", 1, "Time (s)", "Strength (m^2/s)")
# plt.show()
