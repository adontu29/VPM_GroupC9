import numpy as np
import matplotlib.pyplot as plt
import ReadData as rd
import math
from vtriClass import VortexRingInstance

x,y,z,u,v,w,Wx,Wy,Wz,Radius,Group_ID,Viscosity,Viscosity_t = rd.readVortexRingInstance("dataset\Vortex_Ring_DNS_Re7500_0025.vtp")
vtrInstance = VortexRingInstance(x,y,z,u,v,w,Wx,Wy,Wz,Radius,Group_ID,Viscosity,Viscosity_t)

minx = min(vtrInstance.x)
maxx = max(vtrInstance.x)

def findYPlane(vtrInstance, minx, maxx):
    middle = (maxx + minx) / 2
    xs = np.ndarray.tolist(vtrInstance.x)
    wx = np.ndarray.tolist(vtrInstance.wx)
    wy = np.ndarray.tolist(vtrInstance.wy)
    wz = np.ndarray.tolist(vtrInstance.wz)
    for pos in xs:
        vort1 = 0
        numpart1 = 0
        vort2 = 0
        numpart2 = 0
        #absvort = math.sqrt(math.sqrt(wx[int(np.where(vtrInstance.y == pos)[0])]**2 + wy[int(np.where(vtrInstance.y == pos)[0])]**2 + wz[int(np.where(vtrInstance.y == pos)[0])]**2))
        absvort = math.sqrt(math.sqrt(wx[xs.index(pos)]**2 + wy[xs.index(pos)]**2 + wz[xs.index(pos)]**2))
        if pos <= middle: 
            vort1 = vort1 + absvort
            numpart1 = numpart1 + 1
        else:
            vort2 = vort2 + absvort
            numpart2 = numpart2 + 1

    if numpart1 == 0 or numpart2 == 0:
        return middle
    
    vort1avg = vort1 / numpart1
    vort2avg = vort2 / numpart2

    if vort1avg > vort2avg:
        maxx = middle
    else:
        minx = middle

   
    findYPlane(vtrInstance, minx, maxx)

print(findYPlane(vtrInstance, minx, maxx))