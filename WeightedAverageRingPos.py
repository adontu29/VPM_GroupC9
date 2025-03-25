import vtk
import numpy as np
import math as m
import matplotlib.pyplot as plt
import ReadData as rd
import matplotlib.animation as animation

def calcDist (instance1, instance2):
    return  m.sqrt((instance1[0] - instance2[0])**2 + (instance1[1] - instance2[1])**2 + (instance1[2] - instance2[2])**2)


saffmanPos = []
saffmanPosPlot = []
dt = 25E-3
timeStamps = np.arange(0,1575,25)
Velocity = np.ones(len(timeStamps))
ringRadius = np.ones(len(timeStamps))
nu = np.ones(len(timeStamps))
saffmanVelocity = np.ones(len(timeStamps))
ringPos = []
gamma = np.ones(len(timeStamps))

for i in range(len(timeStamps)):
    zeros = ['', '0', '00', '000', '0000']
    # Debugged: Use zfill(4) instead of manual padding
    stringtime = str(timeStamps[i]).zfill(4)
    #print(stringtime, zeros[4-len(stringtime)])

    # Debugged: Ensure correct file path format
    filename = f'dataset/Vortex_Ring_DNS_Re7500_{stringtime}.vtp'

    try:
        X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance(filename)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        continue

    # Debugged: Properly unpack ring position
    ringRadius[i], ringPos0 = rd.getRingPosRadius(X, Y, Z, Wx, Wy, Wz)

    # Debugged: Ensure `Viscosity` is accessed correctly
    if np.ndim(Viscosity) == 1:
        nu[i] = Viscosity[0]
    else:
        nu[i] = Viscosity[1][0]

    # Debugged: Convert ringPos to array for safer indexing
    ringPos.append(np.array(ringPos0))

    # Debugged: Ensure `gamma[i]` is properly computed
    strengthMagnitude = np.sqrt(Wx ** 2 + Wy ** 2 + Wz ** 2)
    gamma[i] = np.sum(strengthMagnitude)

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
        saffmanVelocity[i] = (gamma[i]/(4*np.pi*ringRadius[i]))*(np.log(4*ringRadius[i] / (np.sqrt(nu[i] * timeStamps[i]/1000) + eps)) -0.558 - 3.6716 * nu[i] * timeStamps[i]/1000 / (ringRadius[i] ** 2))
        saffmanPos.append((saffmanVelocity[i]+saffmanVelocity[i-1])/2 * dt)
        saffmanPosPlot.append(sum(saffmanPos))
SaffmanPosInt = sum(saffmanPos)
Error = (SaffmanPosInt - ringPos[-1][0])/SaffmanPosInt

plt.subplot(121)
fig = plt.figure()
ax = plt.axes()
numVel = ax.plot(timeStamps, Velocity, 'b-')
safVel = ax.plot(timeStamps[1:len(timeStamps)-1], saffmanVelocity[1:len(timeStamps)-1], 'r-')
plt.plot(timeStamps[1:], saffmanPosPlot, 'k-')
plt.plot(timeStamps,ringPos, 'k--')

plt.subplot(122)
Energy = 0.5 * Velocity ** 2
plt.plot(timeStamps, Energy, 'k-')
plt.scatter(timeStamps, Energy, c="red", edgecolors='black')


plt.show()

print(f"gamma[{i}] =", gamma[i])
print(f"ringRadius[{i}] =", ringRadius[i])
print(f"nu[{i}] =", nu[i])
print(f"timeStamps[{i}] =", timeStamps[i] / 1000)
print(f"saffmanVelocity[{i}] =", saffmanVelocity[i])
print(Error)
print(SaffmanPosInt)
print(ringPos[-1][0])
# Creating the Animation object
