import sys

import numpy as np
import matplotlib.pyplot as plt
import Test.ReadData as rd
import math
from vtriClass import VortexRingInstance
import vtk

def calcDist (instance1, instance2):
    return math.sqrt((instance1[0] - instance2[0])**2 + (instance1[1] - instance2[1])**2 + (instance1[2] - instance2[2])**2)

#load an instance of the dataset and create a VortexRingInstance object for it
x,y,z,u,v,w,Wx,Wy,Wz,Radius,Group_ID,Viscosity,Viscosity_t = rd.readVortexRingInstance(
    "../dataset/Vortex_Ring_DNS_Re7500_0025.vtp")

print(sys.path)
vtrInstance = VortexRingInstance(x,y,z,u,v,w,Wx,Wy,Wz,Radius,Group_ID,Viscosity,Viscosity_t)

#find min and max coordinates of the particles along the x axis
minx = min(vtrInstance.x)
maxx = max(vtrInstance.x)

#bisection algorithm to find ring core's position along the x-axis, inputs: VortexRingInstance object, min and max x-coordinates, output: x-coordinate of vortex ring core
def findXPlane(vtrInstance, minx, maxx):
    #bisect the domain
    middle = (maxx + minx) / 2

    #convert numpy arrays to lists
    xs = np.ndarray.tolist(vtrInstance.x)
    wx = np.ndarray.tolist(vtrInstance.wx)
    wy = np.ndarray.tolist(vtrInstance.wy)
    wz = np.ndarray.tolist(vtrInstance.wz)

    #create variables for particle numbers and total vorticity for each half of the domain
    vort1 = 0
    numpart1 = 0
    vort2 = 0
    numpart2 = 0

    #iterate through all particle x-positions, calculate the absolute value of the particle's vorticity and increment respective variable
    for pos in xs:
        absvort = math.sqrt(math.sqrt(wy[xs.index(pos)]**2 + wz[xs.index(pos)]**2))
        if pos >= minx and pos <= middle: 
            vort1 = vort1 + absvort
            numpart1 = numpart1 + 1
        elif pos > middle and pos <= maxx:
            vort2 = vort2 + absvort
            numpart2 = numpart2 + 1
    
#if both halfs of domain still contain particles, calculate average vorticities, use domain half with higher avg vorticity and repeat algorithm
    if numpart1 != 0 and numpart2 != 0 and vort1 != 0 and vort2 != 0:
        vort1avg = vort1 / numpart1
        vort2avg = vort2 / numpart2

        if vort1avg > vort2avg:
            return findXPlane(vtrInstance, minx, middle)
        else:
            return findXPlane(vtrInstance, middle, maxx)

    else:
        return middle


#find minimum and maximum radii
maxr = max(vtrInstance.rad)
minr = min(vtrInstance.rad)

#bisection algorithm to find radius of ring core, inputs: VortexRingInstance object, minimum and maximum radius, output: radius of vortex ring core
def findRad(vtrInstance, minr, maxr):
    #bisect domain
    middle = (maxr + minr) / 2

    #convert numpy arrays to lists
    radii = np.ndarray.tolist(vtrInstance.rad)
    wx = np.ndarray.tolist(vtrInstance.wx)
    wy = np.ndarray.tolist(vtrInstance.wy)
    wz = np.ndarray.tolist(vtrInstance.wz)

    #create variables for particle numbers and total vorticity for each half of the domain
    vort1 = 0
    numpart1 = 0
    vort2 = 0
    numpart2 = 0

    #iterate through all particle radii, calculate the absolute value of the particle's vorticity and increment respective variable
    for radius in radii:
        absvort = math.sqrt(math.sqrt(wy[radii.index(radius)]**2 + wz[radii.index(radius)]**2))
        if radius >= minr and radius <= middle: 
            vort1 = vort1 + absvort
            numpart1 = numpart1 + 1
        elif radius > middle and radius <= maxr:
            vort2 = vort2 + absvort
            numpart2 = numpart2 + 1
    
    #if both halfs of domain still contain particles, calculate average vorticities, use domain half with higher avg vorticity and repeat algorithm
    if numpart1 > 100 and numpart2 > 100:
        vort1avg = vort1 / numpart1
        vort2avg = vort2 / numpart2

        if vort1avg > vort2avg:
            return findRad(vtrInstance, minr, middle)
        else:
             return findRad(vtrInstance, middle, maxr)

    else:
        return middle
#print(findRad(vtrInstance, minr, maxr))
#todo: plot core radius over time using this function
timeStamps = np.arange(0,1575,25)
Velocity = np.ones(len(timeStamps))
ringRadius = np.ones(len(timeStamps))
nu = np.ones(len(timeStamps))
saffmanVelocity = np.ones(len(timeStamps))
ringPos = []
gamma = np.ones(len(timeStamps))

ttab = []
rtab = []
xtab = []
vtab = []

for i in range(len(timeStamps)):
    zeros = ['', '0', '00', '000', '0000']
    # Debugged: Use zfill(4) instead of manual padding
    stringtime = str(timeStamps[i]).zfill(4)
    #print(stringtime, zeros[4-len(stringtime)])

    # Debugged: Ensure correct file path format
    filename = f'../dataset/Vortex_Ring_DNS_Re7500_{stringtime}.vtp'


    try:
        X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance(filename)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        continue

    # Debugged: Properly unpack ring position
    ringRadius[i], ringPos0 = rd.getRingPosRadius(X, Y, Z, Wx, Wy, Wz)
    vtrInstance = VortexRingInstance(X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t)
    maxr = max(vtrInstance.rad)
    minr = min(vtrInstance.rad)
    minx = min(vtrInstance.x)
    maxx = max(vtrInstance.x)

    #plotting of x-plane function


    # Debugged: Ensure `Viscosity` is accessed correctly
    if np.ndim(Viscosity) == 1:
        nu[i] = Viscosity[0]
    else:
        nu[i] = Viscosity[1][0]

    ttab.append(float(stringtime)/1000)
    rtab.append(findRad(vtrInstance,minr,maxr))
    xtab.append(findXPlane(vtrInstance,minx,maxx))
    #vtab.append()

    for i in range(len(ttab)-1):
        vtab.append((xtab[i+1]-xtab[i])/0.025)

    # Debugged: Convert ringPos to array for safer indexing
    ringPos.append(np.array(ringPos0))

    # Debugged: Ensure `gamma[i]` is properly computed
    strengthMagnitude = np.sqrt(Wx ** 2 + Wy ** 2 + Wz ** 2)
    gamma[i] = np.sum(strengthMagnitude)
    print(stringtime)
    #print(findRad(vtrInstance,minr,maxr))
# Debugged: Ensure ringPos is properly structured
ringPos = np.array(ringPos)

#plotting
#plt.plot(ttab,rtab)
plt.plot(ttab,vtab)

plt.show()


#to-do thursday:
#plot the velocity graph

#print(f"gamma[{i}] =", gamma[i])
#print(f"ringRadius[{i}] =", ringRadius[i])
#print(f"nu[{i}] =", nu[i])
#print(f"timeStamps[{i}] =", timeStamps[i] / 1000)
#print(f"saffmanVelocity[{i}] =", saffmanVelocity[i])