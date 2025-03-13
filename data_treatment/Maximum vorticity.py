#import the needed modules
import numpy as np
import ReadData

#take the vorticity arrays and positions
omega_x = ReadData.Wx
omega_y = ReadData.Wy
omega_z = ReadData.Wz
x = ReadData.X
y = ReadData.Y
z = ReadData.Z

#take magnitude of only y and z components
omega_magnitude = np.sqrt(omega_y**2 + omega_z**2)

#find the maximum and its location
max_omega = np.max(omega_magnitude)
idx_max_omega = np.where(omega_magnitude == max_omega)

#Find the radius using the coordinates
r_max_omega = np.sqrt(y[idx_max_omega]**2 + z[idx_max_omega]**2)
ring_radius  = r_max_omega

print(ring_radius)