import vtk
import numpy as np
import matplotlib.pyplot as plt
import ReadData as rd
X0,Y0,Z0,U0,V0,W0,Wx0,Wy0,Wz0,Radius0,Group_ID0,Viscosity0,Viscosity_t0 = rd.readVortexRingInstance('dataset/Vortex_Ring_DNS_Re7500_0000.vtp')

timeStamps = np.arange(0,1575,25)
for i in timeStamps:
    zeros = ['', '0', '00', '000', '0000']
    stringtime = str(i)
    print(stringtime, 4-len(stringtime))


# plt.ion()
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# p = ax.scatter(X, Y, Z, c=Strength_magnitude, marker='o')
# fig.colorbar(p)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_aspect('equal', 'box')

# plt.show()



    
