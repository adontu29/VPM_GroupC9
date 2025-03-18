import numpy as np
import matplotlib.pyplot as plt
import ReadData as rd
import math
from vtriClass import VortexRingInstance

x,y,z,u,v,w,Wx,Wy,Wz,Radius,Group_ID,Viscosity,Viscosity_t = rd.readVortexRingInstance("dataset\Vortex_Ring_DNS_Re7500_0025.vtp")
vtrInstance = VortexRingInstance(x,y,z,u,v,w,Wx,Wy,Wz,Radius,Group_ID,Viscosity,Viscosity_t)

minx = min(vtrInstance.x)
maxx = max(vtrInstance.x)

def findXPlane(vtrInstance, minx, maxx):
    middle = (maxx + minx) / 2
    xs = np.ndarray.tolist(vtrInstance.x)
    wx = np.ndarray.tolist(vtrInstance.wx)
    wy = np.ndarray.tolist(vtrInstance.wy)
    wz = np.ndarray.tolist(vtrInstance.wz)

    print("Positions:")
    print(minx)
    print(maxx)
    print(middle)

    vort1 = 0
    numpart1 = 0
    vort2 = 0
    numpart2 = 0

    for pos in xs:
        #absvort = math.sqrt(math.sqrt(wx[int(np.where(vtrInstance.y == pos)[0])]**2 + wy[int(np.where(vtrInstance.y == pos)[0])]**2 + wz[int(np.where(vtrInstance.y == pos)[0])]**2))
        absvort = math.sqrt(math.sqrt(wx[xs.index(pos)]**2 + wy[xs.index(pos)]**2 + wz[xs.index(pos)]**2))
        if pos >= minx and pos <= middle: 
            vort1 = vort1 + absvort
            numpart1 = numpart1 + 1
        elif pos > middle and pos <= maxx:
            vort2 = vort2 + absvort
            numpart2 = numpart2 + 1

    #print(vort1)
    #print(vort2)

    """"if vort1 == 0:
        middle
        findXPlane(vtrInstance, middle, maxx)
    if vort2 == 0:
        maxx = middle
        findXPlane(vtrInstance, minx, middle)"""
    
    if numpart1 != 0 and numpart2 != 0 and vort1 != 0 and vort2 != 0:
        vort1avg = vort1 / numpart1
        vort2avg = vort2 / numpart2
        print("Average vorticities:")
        print(vort1avg)
        print(vort2avg)

        if vort1avg > vort2avg:
            return findXPlane(vtrInstance, minx, middle)
        else:
             return findXPlane(vtrInstance, middle, maxx)

    #elif numpart1 == 0 or numpart2 == 0 or vort1 == 0 or vort2 == 0:
    
    #print("I am here")
    #print(f"Variable: {middle}, Type: {type(middle).__name__}")
    else:
        return middle

result = findXPlane(vtrInstance, minx, maxx)
print(result)