#import the needed modules
import numpy as np
import ReadData as rd
import math as m
from matplotlib import pyplot as plt

def RadiusVelocityPlotsFromMaxVorticity():
    ringPosLst = []
    ringRadiusLst = []
    X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance('dataset/Vortex_Ring_DNS_Re7500_0000.vtp')
    
    timeStamps = np.arange(25,1575,25)
    Velocity = np.ones(len(timeStamps))
    WMagnitude = np.sqrt(Wy**2 + Wz**2)

    #find the maximum and its location
    MaxW = np.max(WMagnitude)
    IdxMaxW = np.where(WMagnitude == MaxW)

    #Find the radius using the coordinates
    XMaxW, YMaxW, ZMaxW = X[IdxMaxW], Y[IdxMaxW], Z[IdxMaxW]
    ringPos0 = tuple([XMaxW, YMaxW, ZMaxW])
    ringRadius0 = np.sqrt(Y[IdxMaxW]**2 + Z[IdxMaxW]**2)

    ringPosLst.append(ringPos0)
    ringRadiusLst.append(ringRadius0)
                    

    for i in range(len(timeStamps)):
        zeros = ['', '0', '00', '000', '0000']
        stringtime = str(timeStamps[i])
        print(stringtime, zeros[4-len(stringtime)])

        X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance('dataset/Vortex_Ring_DNS_Re7500_' + zeros[4-len(stringtime)] + stringtime + '.vtp')

        #take magnitude of only y and z components
        WMagnitude = np.sqrt(Wy**2 + Wz**2)

        #find the maximum and its location
        MaxW = np.max(WMagnitude)
        IdxMaxW = np.where(WMagnitude == MaxW)
        XMaxW, YMaxW, ZMaxW= X[IdxMaxW], Y[IdxMaxW], Z[IdxMaxW]
        ringRadius = np.sqrt(Y[IdxMaxW]**2 + Z[IdxMaxW]**2)
        ringPos = tuple([XMaxW, YMaxW, ZMaxW])
        Velocity[i] = m.sqrt((ringPos[0] - ringPos0[0])**2 + (ringPos[1] - ringPos0[1])**2 + (ringPos[2] - ringPos0[2])**2)/timeStamps[i]*1000
        ringPosLst.append(ringPos)
        ringRadiusLst.append(ringRadius)
        ringPos0 = ringPos
        ringRadius0 = ringRadius
    fig = plt.figure()
    ax = plt.axes()

    line, = ax.plot(timeStamps, Velocity, 'b-')
    plt.show()
    return(ringRadiusLst,ringPosLst)

ringRadiusLst, ringPosLst = RadiusVelocityPlotsFromMaxVorticity()





