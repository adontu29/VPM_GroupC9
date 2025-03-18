#import the needed modules
import numpy as np
import ReadData as rd
import math as m

def radius_from_max_vorticity(filename):
    # get starting conditions
    X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance('dataset/Vortex_Ring_DNS_Re7500_0000.vtp')
    ringRadius0, ringPos0 = rd.getRingPosRadius(X,Y,Z,Wx,Wy,Wz)
    timeStamps = np.arange(25,1575,25)
    Velocity = np.ones(len(timeStamps))
    W_magnitude = np.sqrt(Wy**2 + Wz**2)

    #find the maximum and its location
    max_W = np.max(W_magnitude)
    idx_max_W = np.where(W_magnitude == max_W)

    #Find the radius using the coordinates
    X_max_W, Y_max_W, Z_max_W = X[idx_max_W], Y[idx_max_W], Z[idx_max_W]
    ring_radius0 = np.sqrt(Y[idx_max_W]**2 + Z[idx_max_W]**2)
    

    for i in range(len(timeStamps)):
        zeros = ['', '0', '00', '000', '0000']
        stringtime = str(timeStamps[i])
        print(stringtime, zeros[4-len(stringtime)])

        X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance('dataset/Vortex_Ring_DNS_Re7500_' + zeros[4-len(stringtime)] + stringtime + '.vtp')

        #take magnitude of only y and z components
        W_magnitude = np.sqrt(Wy**2 + Wz**2)

        #find the maximum and its location
        max_W = np.max(W_magnitude)
        idx_max_W = np.where(W_magnitude == max_W)

        #Find the radius using the coordinates
        X_max_W, Y_max_W, Z_max_W = X[idx_max_W], Y[idx_max_W], Z[idx_max_W]
        ring_radius = np.sqrt(Y[idx_max_W]**2 + Z[idx_max_W]**2)
        ringRadius, ringPos = rd.getRingPosRadius(X, Y, Z, Wx, Wy, Wz)
        Velocity[i] = m.sqrt((ringPos[0] - ringPos0[0])**2 + (ringPos[1] - ringPos0[1])**2 + (ringPos[2] - ringPos0[2])**2)/timeStamps[i]*1000

    return ring_radius, X_max_W, Y_max_W, Z_max_W


print(radius_from_max_vorticity('dataset\Vortex_Ring_DNS_Re7500_0000.vtp'))
