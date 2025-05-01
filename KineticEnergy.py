import numpy as np
from vtriClass import VortexRingInstance
from ReadData2 import readVortexRingInstance
from numba import jit

@jit
def Energy(x,y,z,u,v,w,wx,wy,wz,radius,group_id,viscosity,viscosity_t ):
    p = np.stack((x, y, z), axis=1)  # p coordinates
    q = p  # q coordinates
    ap = np.stack((wx, wy, wz), axis=1)  # p particle strength
    aq = ap  # q particle strength
    arr = np.zeros((len(p), len(q)))
    sig = radius[0, 0]
    for i in range(len(p)):
        for j in range(i+1, len(q)):
            if j !=i:
                rho = np.linalg.norm(p[i] - q[j])/sig
                arr[i][j] = 1/np.linalg.norm(p[i]-q[j]) * (((2*rho)/(rho**2+1)**(1/2))* np.dot(ap[i],aq[j]) + rho**3/(rho**2+1)**(3/2)*((np.dot((p[i]-q[j]), ap[i]))*(np.dot((p[i]-q[j]), aq[j])))/(np.linalg.norm(p[i]-q[j]))**2 - np.dot(ap[i], aq[j]))
        if i % 25 == 0:
            print(i)
    E = 1/(16*np.pi)*sum(arr)
    return E

x,y,z,u,v,w,wx,wy,wz,radius,group_id,viscosity,viscosity_t = readVortexRingInstance('dataset/Vortex_Ring_DNS_Re7500_0025.vtp')

print('kinetic energy is', Energy(x,y,z,u,v,w,wx,wy,wz,radius,group_id,viscosity,viscosity_t))

