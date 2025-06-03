#import the needed modules
import numpy as np
import ReadData as rd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numba
from numba import jit

degree = 3

def RadiusVelocityPlotsFromMaxVorticity():
# === Initial Conditions ===

    ring_center   = np.array([0.0, 0.0, 0.0]) # m, center of the vortex
    ring_radius   = 1.0              # m, radius of the vortex ring
    ring_strength = 1.0              # m²/s, vortex strength
    ring_thickness = 0.2*ring_radius # m thickness of the vortex ring

    # ==================================================
    # Particle Distribution Setup
    # ==================================================

    Re = 750                                   # Reynolds number
    particle_distance  = 0.22*ring_thickness    # m
    particle_radius    = 0.8*particle_distance**0.5  # m
    particle_viscosity = ring_strength/Re       # m²/s, kinematic viscosity
    time_step_size     = 25 * 3 * particle_distance**2/ring_strength  # s
    n_time_steps       = int( 100*ring_radius**2 / ring_strength / time_step_size)
    max_timesteps = 8600
    no_timesteps = 8600/25+1

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
                Velocity[i] = calcDist(ringPosLst[i+1],ringPosLst[i])/(time_step_size*25) # forward difference formula
            elif (i==len(timeStamps)-1):
                Velocity[i] = calcDist(ringPosLst[i],ringPosLst[i-1])/(time_step_size*25) # backward difference formula
            else:
                Velocity[i] = calcDist(ringPosLst[i+1],ringPosLst[i-1])/(time_step_size*50) # central difference formula

    return(ringRadiusLst,ringPosLst,Velocity,timeStamps)

def regressionM(X, y, M):
    X = np.array(X).reshape(-1, 1)
    poly = PolynomialFeatures(degree=M)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    return model, poly

ringRadiusLst, ringPosLst, Velocity, timeStamps = RadiusVelocityPlotsFromMaxVorticity()
print((ringPosLst))
print(Velocity)

model, poly = regressionM(timeStamps, Velocity, degree)

# Predict values using the model
X_plot = np.array(timeStamps).reshape(-1, 1)
X_poly_plot = poly.transform(X_plot)
y_pred = model.predict(X_poly_plot)

# Plotting
fig = plt.figure()
ax = plt.axes()
ax.scatter(timeStamps, Velocity, label='Original Data')
ax.plot(timeStamps, y_pred, 'r-', label='nth Degree Polynomial Fit (scikit-learn)')
ax.legend()
plt.show()