import numpy as np
import ReadData as rd
from vtriClass import VortexRingInstance

"""def calcEnstrophy(ringInstance):
    summing = 0
    for i in range(len(ringInstance.x)):
        print(i)
        for j in range(i+1, len(ringInstance.x)):
            pos1 = np.array([ringInstance.x[i], ringInstance.y[i], ringInstance.z[i]])
            pos2 = np.array([ringInstance.x[j], ringInstance.y[j], ringInstance.z[j]])
            rho = np.linalg.norm(pos1 - pos2) / ringInstance.radius[1]

            strength1 = np.array([ringInstance.wx[i], ringInstance.wy[i], ringInstance.wz[i]])
            strength2 = np.array([ringInstance.wx[j], ringInstance.wy[j], ringInstance.wz[j]])

            factor1 = (5 - rho**2 * (rho**2 + 7/2)) / (rho**2 + 1)**(7/2)
            factor2 = 3 * (rho**2 * (rho**2 + 9/2) + 7/2) / (rho**2 + 1)**(9/2)

            summand = (factor1 * np.dot(strength1, strength2) + factor2 * np.dot((pos1 - pos2), strength1) * np.dot((pos1 - pos2), strength2)) / ringInstance.radius[1]**3
        
            summing = summing + summand
            print(summing)

    summing = summing / (4 * np.pi)
    return summing"""

def calcEnstrophy_vec(ringInstance):
    x, y, z = ringInstance.x, ringInstance.y, ringInstance.z
    wx, wy, wz = ringInstance.wx, ringInstance.wy, ringInstance.wz
    radius = ringInstance.radius[1]

    positions = np.stack((x, y, z), axis=1)
    strengths = np.stack((wx, wy, wz), axis=1)

    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    rho2 = r2 / radius**2

    factor1 = (5 - rho2 * (rho2 + 3.5)) / (rho2 + 1)**3.5
    factor2 = 3 * (rho2 * (rho2 + 4.5) + 3.5) / (rho2 + 1)**4.5

    dot_strength = strengths @ strengths.T
    dot_diff_strength1 = np.einsum('ijk,ik->ij', diff, strengths)
    dot_diff_strength2 = np.einsum('ijk,jk->ij', diff, strengths)

    summand = (factor1 * dot_strength + factor2 * dot_diff_strength1 * dot_diff_strength2) / radius**3
    np.fill_diagonal(summand, 0)

    return np.sum(summand) / (4 * np.pi)


x,y,z,u,v,w,Wx,Wy,Wz,Radius,Group_ID,Viscosity,Viscosity_t = rd.readVortexRingInstance(
    "dataset/Vortex_Ring_DNS_Re7500_0025.vtp")

vtrInstance = VortexRingInstance(x,y,z,u,v,w,Wx,Wy,Wz,Radius,Group_ID,Viscosity,Viscosity_t)
        
print(calcEnstrophy_vec(vtrInstance))