import vtk
import numpy as np
import math as m
import matplotlib.pyplot as plt
import ReadData as rd
import matplotlib.animation as animation

def calcDist (instance1, instance2):
    return  m.sqrt((instance1[0] - instance2[0])**2 + (instance1[1] - instance2[1])**2 + (instance1[2] - instance2[2])**2)


timeStamps = np.arange(0,1575,25)
Velocity = np.ones(len(timeStamps))
ringRadius = np.ones(len(timeStamps))
ringPos = []
gamma = np.ones(len(timeStamps))

for i in range(len(timeStamps)):
    zeros = ['', '0', '00', '000', '0000']
    stringtime = str(timeStamps[i])
    #print(stringtime, zeros[4-len(stringtime)])

    X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance('dataset/Vortex_Ring_DNS_Re7500_' + zeros[4-len(stringtime)] + stringtime + '.vtp')
    ringRadius[i], ringPos0 = rd.getRingPosRadius(X, Y, Z, Wx, Wy, Wz)
    ringPos.append(ringPos0)
    strengthMagnitude = np.sqrt(Wx**2+Wy**2+Wz**2)
    gamma[i] = np.sum(strengthMagnitude)

for i in range(len(timeStamps)):
    if (i==0):
        Velocity[i] = calcDist(ringPos[i+1],ringPos[i])/(timeStamps[i+1]-timeStamps[i])*1000
    elif (i==len(timeStamps)-1):
        Velocity[i] = calcDist(ringPos[i],ringPos[i-1])/(timeStamps[i]-timeStamps[i-1])*1000
    else:
        Velocity[i] = calcDist(ringPos[i+1],ringPos[i-1])/(timeStamps[i+1]-timeStamps[i-1])*1000


fig = plt.figure()
ax = plt.axes()
line = ax.plot(timeStamps, Velocity, 'b-')
plt.show()

# Creating the Animation object









print("heyyy")
