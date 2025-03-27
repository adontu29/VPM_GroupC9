import numpy as np
import math as m



def LinearImpulse(CirculationArray, ViscousityArea):

    TimeStamps = np.arange(0,1575,25) / 1000
    LArray = (2*np.array(ViscousityArea)*TimeStamps)**0.5
    theta = 1 / LArray

    gamma0_array = np.array(CirculationArray) / (1 - np.exp(-(theta**2) / 2))

    R0 = 1
    Impulses = m.pi * gamma0_array * (R0**2)  
    
    return Impulses 

Array1 = [0,1,13,2,13]
Array2 = [0,1,2,3,4]

print(LinearImpuls(Array1, Array2))