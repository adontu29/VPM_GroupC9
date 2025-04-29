import numpy as np
from vtriClass import VortexRingInstance
from ReadData2 import readVortexRingInstance


vring = VortexRingInstance(readVortexRingInstance('filename'))

p = [vring.x, vring.y, vring.z]         #p coordinates
q = p                                   #q coordinates
ap = [vring.wx, vring.wy, vring.wz]    #p particle strength
aq = ap                                 #q particle strength
sig = 5                                 #smoothing/size?

arr = np.zeros((len(p), len(q)))

for i in range(len(p)):
    for j in range(len(q)):
        if j !=i:
            rho = np.abs(p[i]-q[j])/sig
            arr[i][j] = 1/np.abs(p[i]-q[j]) * (((2*rho)/(rho**2+1)**(1/2))* np.dot(ap[i],aq[j]) + rho**3/(rho**2+1)**(3/2)*((np.dot((p[i]-q[j]), ap[i]))*(np.dot((p[i]-q[j]), aq[j])))/(np.abs(p[i]-q[j]))**2 - np.dot(ap[i], aq[j]))

E = 1/(16*np.pi)*sum(arr)
