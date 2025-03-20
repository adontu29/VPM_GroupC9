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
nu = np.ones(len(timeStamps))
saffmanVelocity = np.ones(len(timeStamps))
saffmanSimplified = np.zeros(len(timeStamps))
ringPos = []
gamma = np.ones(len(timeStamps))

for i in range(len(timeStamps)):
    zeros = ['', '0', '00', '000', '0000']
    stringtime = str(timeStamps[i])
    #print(stringtime, zeros[4-len(stringtime)])

    X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance('dataset/Vortex_Ring_DNS_Re7500_' + zeros[4-len(stringtime)] + stringtime + '.vtp')
    ringRadius[i], ringPos0 = rd.getRingPosRadius(X, Y, Z, Wx, Wy, Wz)
    nu[i] = Viscosity[1][0]
    ringPos.append(ringPos0)
    strengthMagnitude = np.sqrt(Wx**2+Wy**2+Wz**2)
    gamma[i] = nu[i]*7500

timeStampsSec = timeStamps/1000
for i in range(len(timeStampsSec)):
    if (i==0):
        Velocity[i] = calcDist(ringPos[i+1],ringPos[i])/(timeStampsSec[i+1]-timeStampsSec[i])
    elif (i==len(timeStampsSec)-1):
        Velocity[i] = calcDist(ringPos[i],ringPos[i-1])/(timeStampsSec[i]-timeStampsSec[i-1])
    else:
        Velocity[i] = calcDist(ringPos[i+1],ringPos[i-1])/(timeStampsSec[i+1]-timeStampsSec[i-1])
    if (i!=0):
        saffmanVelocity[i] = gamma[i]/(4*np.pi*ringRadius[i])*(np.log(4*ringRadius[i] / np.sqrt(nu[i] * timeStampsSec[i]))-0.558 - 3.6716 * nu[i] * timeStampsSec[i] / ringRadius[i] ** 2)
        #saffmanSimplified[i] = gamma[i]/(4*np.pi*ringRadius[i])*(np.log(4*ringRadius[i] / np.sqrt(nu[i] * timeStamps[i]/1000))-0.558)
fig = plt.figure(1)
ax = plt.axes()
#vorticity_magnitude = (ring_strength/(np.pi*ring_thickness**2)) * np.exp(-radial_distance_to_core**2/ring_thickness**2)
numVel = ax.plot(timeStamps/1000, Velocity, 'b-')
safVel = ax.plot(timeStamps[1:len(timeStamps)-1]/1000, saffmanVelocity[1:len(timeStamps)-1], 'r-')
ax.legend(['VPM Velocity'])

fig2 = plt.figure(2)
ax = plt.axes()
radRing = ax.plot(timeStamps/1000, ringRadius, 'g-')
ax.legend(['Ring Radius'])

fig3 = plt.figure(3)
ax = plt.axes()
radRing = ax.plot(timeStamps/1000, ringPos, 'g-')
ax.legend(['Ring Position'])
plt.show()
# Creating the Animation object
print(gamma)