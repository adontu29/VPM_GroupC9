import numpy as np
import matplotlib.pyplot as plt
import ReadData as rd
import math

X,Y,Z,U,V,W,Wx,Wy,Wz,Radius,Group_ID,Viscosity,Viscosity_t = rd.readVortexRingInstance('dataset/Vortex_Ring_DNS_Re7500_0000.vtp')

print(X)