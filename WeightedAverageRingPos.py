import numpy as np
import math as m
import matplotlib.pyplot as plt
import ReadData as rd


def calcDist (instance1, instance2):
    return  m.sqrt((instance1[0] - instance2[0])**2 + (instance1[1] - instance2[1])**2 + (instance1[2] - instance2[2])**2)

ring_center     = np.array([0.0, 0.0, 0.0])   # m, center of the vortex ring
ring_radius     = 1.0               # m, radius of the vortex ring
ring_strength   = 1.0               # m²/s, vortex strength
ring_thickness  = 0.2*ring_radius   # m, thickness of the vortex ring


### Particle Distribution Setup
Re = 7500                                   # Reynolds number
particle_distance  = 0.25*ring_thickness    # m
particle_radius    = 0.8*particle_distance**0.5  # m
particle_viscosity = ring_strength/Re       # m²/s, kinematic viscosity
time_step_size     = 25*5 * particle_distance**2/ring_strength  # s
n_time_steps       = int( 20*ring_radius**2 / ring_strength / time_step_size)


timeStampsNames = np.arange(0,1575,25)
timeStamps = np.arange(0, (1575/25)*time_step_size, time_step_size)

print(len(timeStampsNames))
print(len(timeStamps))
print(time_step_size)


Velocity = np.ones(len(timeStamps))
ringRadius = np.ones(len(timeStamps))
nu = np.ones(len(timeStamps))
saffmanVelocity = np.ones(len(timeStamps))
ringStrength = np.ones(len(timeStamps))
ringStrengthGrid = [4.98241,4.11510,2.10322,0.86489,1.00185,1.05919,0.86324,0.73651,0.64519,0.56933,0.50394,0.44722]
saffManVelocityGrid = np.ones(len(ringStrengthGrid))
ringPos = []
ringPosPlot = []
ringCoreRadius = np.zeros(len(timeStamps))
gamma = np.ones(len(timeStamps))

for i in range(len(timeStamps)):
    stringtime = str(timeStampsNames[i]).zfill(4)
    # Debugged: Ensure correct file path format
    filename = f'dataset/Vortex_Ring_DNS_Re7500_{stringtime}.vtp'

    try:
        X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance(filename)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        continue

    # Debugged: Properly unpack ring position
    ringRadius[i], ringPos0 = rd.getRingPosRadius(X, Y, Z, Wx, Wy, Wz)
    ringCoreRadius[i] = rd.getRingCoreRadius(X,Y,Z,Wx,Wy,Wz,ringPos0)

    # Debugged: Ensure `Viscosity` is accessed correctly
    if np.ndim(Viscosity) == 1:
        nu[i] = Viscosity[0]
    else:
        nu[i] = Viscosity[1][0]

    # Debugged: Convert ringPos to array for safer indexing
    ringPos.append(np.array(ringPos0))
    ringPosPlot.append(ringPos0[0])

    # Debugged: Ensure `gamma[i]` is properly computed
    strengthMagnitude = np.sqrt(Wx ** 2 + Wy ** 2 + Wz ** 2)
    gamma[i] = np.sum(strengthMagnitude)
    ringStrength[i]=np.sum(strengthMagnitude)
    #ringStrength[i] = rd.getRingStrength(X, Y, Z, Wx, Wy, Wz, ringPos[i], Radius[1][0], ringCoreRadius[i])
# Debugged: Ensure ringPos is properly structured
ringPos = np.array(ringPos)

for i in range(len(timeStamps)):
    if (i==0):
        Velocity[i] = calcDist(ringPos[i+1],ringPos[i])/time_step_size
    elif (i==len(timeStamps)-1):
        Velocity[i] = calcDist(ringPos[i],ringPos[i-1])/time_step_size
    else:
        Velocity[i] = calcDist(ringPos[i+1],ringPos[i-1])/time_step_size/2
    if (i!=0):
        eps = 1e-8
        saffmanVelocity[i] = (ring_strength/(4*np.pi*ringRadius[i]))*(np.log(4*ringRadius[i] / (np.sqrt(nu[i] * timeStamps[i]))) -0.558 - 3.6716*nu[i]*timeStamps[i]/ringRadius[i])

# for i in range(len(ringStrengthGrid)):
#     saffManVelocityGrid[i] = (ringStrengthGrid[i] / (4 * np.pi * ringRadius[i])) * (np.log(4 * ringRadius[i] / (np.sqrt(nu[i] * timeStamps[i]) + eps)) - 0.558 - 3.6716 * nu[i] *
#                 timeStamps[i]  / (ringRadius[i] ** 2))

fig1 = plt.figure(1)
ax = plt.axes()
numVel = ax.plot(timeStamps, Velocity, 'b-')
safVel = ax.plot(timeStamps[1:len(timeStamps)-1], saffmanVelocity[1:len(timeStamps)-1], 'r-')
# safVelGrid = ax.plot(timeStamps[1:(len(ringStrengthGrid))],saffManVelocityGrid[1:(len(ringStrengthGrid)+1)],'g-'),
fig2 = plt.figure(2)
ax = plt.axes()
ax.plot(timeStamps, ringCoreRadius, 'b-')

fig3 = plt.figure(3)
ax = plt.axes()
ax.plot(timeStamps, ringRadius, 'b-')

plt.show()

print(saffmanVelocity)
print(Velocity)
print(f"gamma[{i}] =", gamma[i])
print(f"nu[{i}] =", nu[i])
print(f"timeStamps[{i}] =", timeStamps[i])
print(f"saffmanVelocity[{i}] =", saffmanVelocity[i])



