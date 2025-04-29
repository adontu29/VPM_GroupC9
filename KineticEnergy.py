import numpy as np

p = [1,2,3]     #p coordinates
q = [4,5,6]     #q coordinates
ap = [1,2,3]    #p particle strength
aq = [4,5,6]    #q particle strength
sig = 5

arr = np.zeros((len(p), len(q)))

for i in range(len(p)):
    for j in range(len(q)):
        rho = np.abs(p[i]-q[j])/sig
        arr[i][j] = 1/np.abs(p[i]-q[j]) * (((2*rho)/(rho**2+1)**(1/2))* np.dot(ap[i],aq[j]) + rho**3/(rho**2+1)**(3/2)*((np.dot((p[i]-q[j]), ap[i]))*(np.dot((p[i]-q[j]), aq[j])))/(np.abs(p[i]-q[j]))**2 - np.dot(ap[i], aq[j]))

E = 1/(16*np.pi)*sum(arr)
