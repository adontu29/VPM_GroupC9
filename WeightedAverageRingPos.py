import vtk
import numpy as np
import math as m
import matplotlib.pyplot as plt
import ReadData as rd
import matplotlib.animation as animation

X,Y,Z,U,V,W,Wx,Wy,Wz,Radius,Group_ID,Viscosity,Viscosity_t = rd.readVortexRingInstance('dataset/Vortex_Ring_DNS_Re7500_0000.vtp')
ringRadius0, ringPos0 = rd.getRingPosRadius(X,Y,Z,Wx,Wy,Wz)
Strength_magnitude = np.sqrt(np.square(Wx) + np.square(Wy) + np.square(Wz))




timeStamps = np.arange(25,1575,25)
Velocity = np.ones(len(timeStamps))
for i in range(len(timeStamps)):
    zeros = ['', '0', '00', '000', '0000']
    stringtime = str(timeStamps[i])
    print(stringtime, zeros[4-len(stringtime)])

    X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance('dataset/Vortex_Ring_DNS_Re7500_' + zeros[4-len(stringtime)] + stringtime + '.vtp')
    ringRadius, ringPos = rd.getRingPosRadius(X, Y, Z, Wx, Wy, Wz)
    Velocity[i] = m.sqrt((ringPos[0] - ringPos0[0])**2 + (ringPos[1] - ringPos0[1])**2 + (ringPos[2] - ringPos0[2])**2)/timeStamps[i]*1000


fig = plt.figure()
ax = plt.axes()

line, = ax.plot(timeStamps, Velocity, 'b-')
plt.show()

# Creating the Animation object







    

print("heyyy")
