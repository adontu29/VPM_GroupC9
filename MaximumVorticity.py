#import the needed modules
import numpy as np
import ReadData as rd
from matplotlib import pyplot as plt


def RadiusVelocityPlotsFromMaxVorticity():
    ring_center     = np.array([0.0, 0.0, 0.0])   # m, center of the vortex ring
    ring_radius     = 1.0               # m, radius of the vortex ring
    ring_strength   = 1.0               # m²/s, vortex strength
    ring_thickness  = 0.2*ring_radius   # m, thickness of the vortex ring


    ### Particle Distribution Setup

    Re = 7500                                   # Reynolds number
    particle_distance  = 0.25*ring_thickness    # m
    particle_radius    = 0.8*particle_distance**0.5  # m
    particle_viscosity = ring_strength/Re       # m²/s, kinematic viscosity
    time_step_size     = 5 * particle_distance**2/ring_strength  # s
    n_time_steps       = int( 20*ring_radius**2 / ring_strength / time_step_size)

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
    timeStampMultiplyer = 1
    timeStamps_names = np.arange(25*timeStampMultiplyer,1575,25*timeStampMultiplyer)
    timeStamps = timeStamps_names * time_step_size
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
        stringtime = str(timeStamps_names[i])

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
                Velocity[i] = calcDist(ringPosLst[i+1],ringPosLst[i])/(timeStamps[i+1]-timeStamps[i]) # forward difference formula
            elif (i==len(timeStamps)-1):
                Velocity[i] = calcDist(ringPosLst[i],ringPosLst[i-1])/(timeStamps[i]-timeStamps[i-1]) # backward difference formula
            else:
                Velocity[i] = calcDist(ringPosLst[i+1],ringPosLst[i-1])/(timeStamps[i+1]-timeStamps[i-1]) # central difference formula

    return(ringRadiusLst,ringPosLst,Velocity,timeStamps)

ringRadiusLst, ringPosLst, Velocity, timeStamps = RadiusVelocityPlotsFromMaxVorticity()
print((ringPosLst))
print(Velocity)

def regressionM(X, y, M):
    coeffs = np.polyfit(X, y, M)
    return(coeffs)

def function(X, a, b, c, d, e, f):
    y = a * X ** 5 + b * X ** 4 + c * X ** 3 + d * X ** 2 + e * X + f
    return y

coeffs = regressionM(timeStamps, Velocity, 5)
a, b, c, d, e, f = coeffs

fig = plt.figure()
ax = plt.axes()

line, = ax.plot(timeStamps, Velocity, 'b-')
line, = ax.plot(timeStamps, function(timeStamps,a,b,c,d,e,f), 'r-')

plt.show()
