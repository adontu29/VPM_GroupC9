#import the needed modules
import numpy as np
import ReadData as rd
import math as m
from matplotlib import pyplot as plt

'''
Do pip install for the scikit module
pip install scikit-learn
'''

'''
This funtion looks the particle with the highest vorticity and finds its location.
Because the particle will be located at the core we can approximate the ring velocity by observing the change in x position of said highest vorticity particles
'''

def RadiusVelocityPlotsFromMaxVorticity():
    def calcDist (instance1, instance2):
        h = instance1 - instance2
        return h
    
    ringPosLst = []
    ringRadiusLst = []
    X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance('dataset/Vortex_Ring_DNS_Re7500_0000.vtp')
    
    timeStamps = np.arange(0,1575,25)
    Velocity = np.ones(len(timeStamps))
    WMagnitude = np.sqrt(Wy**2 + Wz**2)

    #find the maximum and its location
    MaxW = np.max(WMagnitude)
    IdxMaxW = np.where(WMagnitude == MaxW)

    #Find the radius using the coordinates
    XMaxW, YMaxW, ZMaxW = X[IdxMaxW][0], Y[IdxMaxW][0], Z[IdxMaxW][0]
    ringPosLst.append(YMaxW)
    ringRadiusLst.append(np.sqrt(Y[IdxMaxW]**2 + Z[IdxMaxW]**2))

                    

    for i in range(len(timeStamps)):
        zeros = ['', '0', '00', '000', '0000']
        stringtime = str(timeStamps[i])


        X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance('dataset/Vortex_Ring_DNS_Re7500_' + zeros[4-len(stringtime)] + stringtime + '.vtp')

        #take magnitude of only y and z comenponents
        WMagnitude = np.sqrt(Wy**2 + Wz**2)

        #find the maximum and its location
        MaxW = np.max(WMagnitude)
        IdxMaxW = np.where(WMagnitude == MaxW)
        XMaxW, YMaxW, ZMaxW = X[IdxMaxW][0], Y[IdxMaxW][0], Z[IdxMaxW][0]
        print(XMaxW)
        #print(YMaxW)
        #print(ZMaxW)
        ringRadiusLst.append(np.sqrt(Y[IdxMaxW]**2 + Z[IdxMaxW]**2))
        ringPosLst.append(XMaxW)

    
    for i in range(len(timeStamps)):
            if (i==0):
                Velocity[i] = calcDist(ringPosLst[i+1],ringPosLst[i])/(timeStamps[i+1]-timeStamps[i])*1000
            elif (i==len(timeStamps)-1):
                Velocity[i] = calcDist(ringPosLst[i],ringPosLst[i-1])/(timeStamps[i]-timeStamps[i-1])*1000
            else:
                Velocity[i] = calcDist(ringPosLst[i+1],ringPosLst[i-1])/(timeStamps[i+1]-timeStamps[i-1])*1000
    fig = plt.figure()
    ax = plt.axes()

    line, = ax.plot(timeStamps, Velocity, 'b-')
    plt.show()
    return(ringRadiusLst,ringPosLst)

ringRadiusLst, ringPosLst = RadiusVelocityPlotsFromMaxVorticity()





