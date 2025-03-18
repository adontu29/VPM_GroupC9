#import the needed modules
import numpy as np
import ReadData

def radius_from_max_vorticity(filename):
    #take the vorticity arrays and positions
    [X,Y,Z,U,V,W,Wx,Wy,Wz,Radius,Group_ID,Viscosity,Viscosity_t] = ReadData.readVortexRingInstance(filename)

    #take magnitude of only y and z components
    W_magnitude = np.sqrt(Wy**2 + Wz**2)

    #find the maximum and its location
    max_W = np.max(W_magnitude)
    idx_max_W = np.where(W_magnitude == max_W)

    #Find the radius using the coordinates
    X_max_W, Y_max_W, Z_max_W = X[idx_max_W], Y[idx_max_W], Z[idx_max_W]
    ring_radius = np.sqrt(Y[idx_max_W]**2 + Z[idx_max_W]**2)

    return ring_radius, X_max_W, Y_max_W, Z_max_W

print(radius_from_max_vorticity('Vortex_Ring_DNS_Re7500_0025.vtp'))