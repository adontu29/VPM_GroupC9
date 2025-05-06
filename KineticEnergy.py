import numpy as np
from vtriClass import VortexRingInstance
from ReadData2 import readVortexRingInstance
from numba import njit,jit

@njit
def getEnergy1(X,Y,Z,Wx,Wy,Wz,radius):
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
                dotprd = np.dot(ap[i],aq[j])
                rho = np.linalg.norm(p[i] - q[j])/sig
                arr[i][j] = 1/np.linalg.norm(diff) * (((2*rho)/(rho**2+1)**(1/2))* dotprd + rho**3/(rho**2+1)**(3/2)*((np.dot((diff), ap[i]))*(np.dot((diff), aq[j])))/(np.linalg.norm(diff))**2 - dotprd)
        if i % 250 == 0:
            print(i)
    E = 1/(16*np.pi)*np.sum(arr)
    return E

@njit
def getEnergy2(X,Y,Z,Wx,Wy,Wz,radius):
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

print('kinetic energy is', getEnergy1(x,y,z,wx,wy,wz,radius))
print('kinetic energy is', getEnergy2(x,y,z,wx,wy,wz,radius))
print('kinetic energy is', getEnergy3(x,y,z,wx,wy,wz,radius))

