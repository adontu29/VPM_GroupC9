import vtk
import numpy as np
import matplotlib.pyplot as plt
from ReadData import readVortexRingInstance

X,Y,Z,U,V,W,Wx,Wy,Wz,Radius,Group_ID,Viscosity,Viscosity_t = readVortexRingInstance('dataset/Vortex_Ring_DNS_Re7500_0000.vtp')

Strength_magnitude = np.sqrt(np.square(Wx) + np.square(Wy) + np.square(Wz))
PositionVector = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
maxStrength = np.max(Strength_magnitude)
Threshold = 0
X_vp = X[Strength_magnitude > maxStrength*Threshold]
Y_vp = Y[Strength_magnitude > maxStrength*Threshold]
Z_vp = Z[Strength_magnitude > maxStrength*Threshold]
Radius = Radius[Strength_magnitude > maxStrength*Threshold]
Wx_vp = Wx[Strength_magnitude > maxStrength*Threshold]
Wy_vp = Wy[Strength_magnitude > maxStrength*Threshold]
Wz_vp = Wz[Strength_magnitude > maxStrength*Threshold]
Strength_magnitude_vp = Strength_magnitude[Strength_magnitude > maxStrength*Threshold]

Strength_total = sum(Strength_magnitude_vp)
X_avg = 0
Y_avg = 0
Z_avg = 0
Radius_avg = 0
for i in range(len(Strength_magnitude_vp)):
    weight = Strength_magnitude_vp[i]
    X_avg += X_vp[i] * weight
    Y_avg += Y_vp[i] * weight
    Z_avg += Z_vp[i] * weight
    Radius_avg += PositionVector[i] * weight

X_avg /= Strength_total; Y_avg /= Strength_total; Z_avg /= Strength_total; Radius_avg /= Strength_total
VortexRingPosition = tuple([X_avg, Y_avg,  Z_avg])

print ("VortexRingPosition:", VortexRingPosition)
print ("VortexRingRadius:", Radius_avg)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
p = ax.scatter(X_vp, Y_vp, Z_vp, c=Strength_magnitude_vp, marker='o')
fig.colorbar(p)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_aspect('equal', 'box')

plt.show()

print("heyyy")
