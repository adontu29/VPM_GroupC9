#import the needed modules
import numpy as np
import pandas as pd
import scipy as sp
import ReadData as rd
import math as m
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

'''
Do pip install for the scikit module
pip install scikit-learn
'''

'''
This funtion looks the particle with the highest vorticity and finds its location.
Because the particle will be located at the core we can approximate the ring velocity by observing the change in x position of said highest vorticity particles
'''

def RadiusVelocityPlotsFromMaxVorticity():

    # calculates the h value for numerical differentiation 
    def calcDist (instance1, instance2):
        h = instance1 - instance2
        return h

    # list definitions
    ringPosLst = []
    ringRadiusLst = []

    #Threshold
    threshold = 0.5

    # reads the 1st datafile and extracts the needed information
    X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance('dataset/Vortex_Ring_DNS_Re7500_0000.vtp')

    # defines the timestamps and sets the length of the velocity array
    timeStampMultiplyer = 2
    timeStamps = np.arange(25*timeStampMultiplyer,1575,25*timeStampMultiplyer)
    Velocity = np.ones(len(timeStamps))

    # obtains the magnitude of the vorticity in the Y and Z direction, as the one in the X direction represents swirl
    WMagnitude = np.sqrt(Wy**2 + Wz**2)

    # finds the maximums and their locations
    MaxW = np.max(WMagnitude)
    IdxMaxW = np.where(WMagnitude == MaxW)

    #Find the radius using the coordinates
    XMaxW, YMaxW, ZMaxW = X[IdxMaxW][0], Y[IdxMaxW][0], Z[IdxMaxW][0]
    ringPosLst.append(YMaxW)
    ringRadiusLst.append(np.sqrt(Y[IdxMaxW]**2 + Z[IdxMaxW]**2))

    for i in range(len(timeStamps)):
        # loops through the files
        zeros = ['', '0', '00', '000', '0000']
        stringtime = str(timeStamps[i])

        # reads through the datafiles and extracts the relevant information
        X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance('dataset/Vortex_Ring_DNS_Re7500_' + zeros[4-len(stringtime)] + stringtime + '.vtp')

        # take magnitude of only y and z components
        WMagnitude = np.sqrt(Wy**2 + Wz**2)

        # find the maximums and their locations
        MaxW = np.max(WMagnitude)
        WThreshold = MaxW * threshold
        IdxMaxW = np.where(WMagnitude == MaxW)
        IdxThreshold = np.where(WMagnitude > WThreshold)

        # because an array/list might be created from there being more than one identical particle, a value is chosen to make sure we get a scalar
        XMaxW, YMaxW, ZMaxW = X[IdxMaxW][0], Y[IdxMaxW][0], Z[IdxMaxW][0]

        # the lists are updated
        ringRadiusLst.append(np.sqrt(YMaxW**2 + ZMaxW**2))
        ringPosLst.append(XMaxW)

    for i in range(len(timeStamps)):
            if (i==0):
                Velocity[i] = calcDist(ringPosLst[i+1],ringPosLst[i])/(timeStamps[i+1]-timeStamps[i])*1000 # forward difference formula
            elif (i==len(timeStamps)-1):
                Velocity[i] = calcDist(ringPosLst[i],ringPosLst[i-1])/(timeStamps[i]-timeStamps[i-1])*1000 # backward difference formula
            else:
                Velocity[i] = calcDist(ringPosLst[i+1],ringPosLst[i-1])/(timeStamps[i+1]-timeStamps[i-1])*1000 # central difference formula

    

    return(ringRadiusLst,ringPosLst,Velocity,timeStamps)

ringRadiusLst, ringPosLst, Velocity, timeStamps = RadiusVelocityPlotsFromMaxVorticity()
print((ringPosLst))
print(Velocity)


fig = plt.figure()
ax = plt.axes()

line, = ax.plot(timeStamps, Velocity, 'b-')
plt.show()

'''
What I am trying to do here is to implement some polynomial regression. 
If you run the file you will see the data obtained is quite noisy.
Maybe by doing that we can obtain a more accurate reading of the data.
You can compare with the weighted average method as that is meant to be the most accurate.
'''

# def preprocessing(X,y):
#     X_avg = np.mean(X)
#     y_avg = np.mean(y)
#     return(X,y)

# X,y = preprocessing(timeStamps,Velocity)    

# def linearRegression(XTrain,yTrain):
#     model = LinearRegression()
#     model.fit(XTrain.reshape(-1, 1), yTrain)
#     return(model)

# model = linearRegression(X,y)

# y_pred = model.predict(X)
