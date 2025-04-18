import numpy as np
import math as m



#===Linear Impuls===

















#===Linear Impulse function, For the comparison====
#This function calculates the Linear Impuls for every time step. 
#The input is a array of the Circulation values of the hole vortex ring for every time step
#and the viscousity is constant so can just plug that in the function command.

Viscousity = 0.00013333333333333334

def LinearImpuls(CirculationArray, Viscousity):

    TimeStamps = np.arange(0,1575,25) / 1000
    LArray = (2*np.array(Viscousity)*TimeStamps)**0.5
    theta = 1 / LArray

    gamma0_array = np.array(CirculationArray) / (1 - np.exp(-(theta**2) / 2))

    R0 = 1
    Impulses = m.pi * gamma0_array * (R0**2)  
    
    return Impulses 
