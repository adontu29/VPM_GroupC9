import numpy as np
from vtriClass import VortexRingInstance
from ReadData2 import readVortexRingInstance
from numba import njit,jit


from numba import njit
import numpy as np

@njit
def getEnergy2(X, Y, Z, Wx, Wy, Wz, radius):
    n = len(X)
    sig = radius[0, 0]
    E = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = X[i] - X[j]
            dy = Y[i] - Y[j]
            dz = Z[i] - Z[j]
            norm_sq = dx*dx + dy*dy + dz*dz
            norm = np.sqrt(norm_sq)
            rho = norm / sig

            # ap[i] and aq[j] dot product
            dot_apaq = Wx[i]*Wx[j] + Wy[i]*Wy[j] + Wz[i]*Wz[j]
            
            # dot_diffap and dot_diffaq
            dot_diffap = Wx[i]*dx + Wy[i]*dy + Wz[i]*dz
            dot_diffaq = Wx[j]*dx + Wy[j]*dy + Wz[j]*dz

            term1 = (2*rho)/np.sqrt(rho**2 + 1) * dot_apaq
            term2 = (rho**3)/( (rho**2 + 1)**1.5 ) * (dot_diffap * dot_diffaq) / norm_sq
            contribution = (1.0 / norm) * (term1 + term2 - dot_apaq)
            E += contribution

        # Optional progress update (disable in performance runs)
        if i % 25 == 0:
            print(i)  # or: print(i) if testing outside @njit

    E = E / (16 * np.pi)
    return E


@jit
def getEnergy3(X,Y,Z,Wx,Wy,Wz,radius):
    print("Getting Kinetic Energy ... ")
    p = np.stack((X, Y, Z), axis=1)  # p coordinates
    q = p  # q coordinates
    ap = np.stack((Wx, Wy, Wz), axis=1)  # p particle strength
    aq = ap  # q particle strength
    arr = np.zeros((len(p), len(q)))
    sig = radius[0, 0]
    for i in range(len(p)):
        for j in range(i+1, len(q)):
            if j !=i:
                diff = p[i]-q[j]
                dot_apaq =  ap[i,0] * aq[j,0] + ap[i,1] * aq[j,1] + ap[i,2] * aq[j,2] 
                dot_diffap = ap[i,0] * diff[0] + ap[i,1] * diff[1] + ap[i,2] * diff[2] 
                dot_diffaq = aq[j,0] * diff[0] + aq[j,1] * diff[1] + aq[j,2] * diff[2]  
                rho = np.linalg.norm(p[i] - q[j])/sig
                arr[i][j] = 1/np.linalg.norm(diff) * (((2*rho)/(rho**2+1)**(1/2))* dot_apaq + rho**3/(rho**2+1)**(3/2)*((dot_diffap)*(dot_diffaq))/(np.linalg.norm(diff))**2 - dot_apaq)
        if i % 25 == 0:
            print(i)
    E = 1/(16*np.pi)*np.sum(arr)
    return E

x,y,z,u,v,w,wx,wy,wz,radius,group_id,viscosity,viscosity_t = readVortexRingInstance('dataset/Vortex_Ring_DNS_Re7500_0025.vtp')


print('kinetic energy is', getEnergy2(x,y,z,wx,wy,wz,radius))
print('kinetic energy is', getEnergy3(x,y,z,wx,wy,wz,radius))

