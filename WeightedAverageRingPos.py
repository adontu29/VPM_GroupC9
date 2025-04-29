import numpy as np
import math as m
import matplotlib.pyplot as plt
from Test import ReadData as rd
#import ReadData as rd

def calcDist (instance1, instance2):
    return  m.sqrt((instance1[0] - instance2[0])**2 + (instance1[1] - instance2[1])**2 + (instance1[2] - instance2[2])**2)



timeStamps = np.arange(0,1575,25)
Velocity = np.ones(len(timeStamps))
ringRadius = np.ones(len(timeStamps))
nu = np.ones(len(timeStamps))
saffmanVelocity = np.ones(len(timeStamps))
ringStrength = np.ones(len(timeStamps))
ringPos = []
ringPosPlot = []
ringCoreRadius = np.zeros(len(timeStamps))
gamma = np.ones(len(timeStamps))

for i in range(len(timeStamps)):
    stringtime = str(timeStamps[i]).zfill(4)
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
    ringStrength[i] = rd.getRingStrength(X, Y, Z, Wx, Wy, Wz, ringPos[i], Radius[1][0], ringCoreRadius[i])
# Debugged: Ensure ringPos is properly structured
ringPos = np.array(ringPos)

for i in range(len(timeStamps)):
    if (i==0):
        Velocity[i] = calcDist(ringPos[i+1],ringPos[i])/(timeStamps[i+1]-timeStamps[i])*1000
    elif (i==len(timeStamps)-1):
        Velocity[i] = calcDist(ringPos[i],ringPos[i-1])/(timeStamps[i]-timeStamps[i-1])*1000
    else:
        Velocity[i] = calcDist(ringPos[i+1],ringPos[i-1])/(timeStamps[i+1]-timeStamps[i-1])*1000
    if (i!=0):
        eps = 1e-8
        saffmanVelocity[i] = (ringStrength[i]/(4*np.pi*ringRadius[i]))*(np.log(4*ringRadius[i] / (np.sqrt(nu[i] * timeStamps[i]/1000) + eps)) -0.558 - 3.6716 * nu[i] * timeStamps[i]/1000 / (ringRadius[i] ** 2))


fig1 = plt.figure(1)
ax = plt.axes()
numVel = ax.plot(timeStamps/1000, Velocity, 'b-')
safVel = ax.plot(timeStamps[1:len(timeStamps)-1]/1000, saffmanVelocity[1:len(timeStamps)-1], 'r-')

fig2 = plt.figure(2)
ax = plt.axes()
coreRad = ax.plot(timeStamps/1000, ringCoreRadius, 'b-')

fig3 = plt.figure(3)
ax = plt.axes()
ringRadius = ax.plot(timeStamps/1000, ringRadius, 'b-')

fig4 = plt.figure(4)
ax = plt.axes()
ringStrength = ax.plot(timeStamps/1000, ringStrength, 'b-')


plt.show()


print(f"gamma[{i}] =", gamma[i])
print(f"ringRadius[{i}] =", ringRadius[i])
print(f"nu[{i}] =", nu[i])
print(f"timeStamps[{i}] =", timeStamps[i] / 1000)
print(f"saffmanVelocity[{i}] =", saffmanVelocity[i])

