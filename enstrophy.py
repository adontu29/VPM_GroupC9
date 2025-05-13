import numpy as np
import ReadData2 as rd
from vtriClass import VortexRingInstance
import matplotlib.pyplot as plt

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

    i_upper = np.triu_indices_from(summand, k=1)
    return np.sum(summand[i_upper]) / (4 * np.pi)

DATA_PATH = "dataset2"
FILENAME_TEMPLATE = "Vortex_Ring_{:04d}.vtp"
#TIMESTAMPS = np.arange(25, 1575, 25)  # in steps of 25
TIMESTAMPS = np.arange(25, 8600, 225)

enstrophies = []
times = []

for stamp in TIMESTAMPS:
    print(stamp)
    filename = f"{DATA_PATH}/{FILENAME_TEMPLATE.format(stamp)}"
    x,y,z,u,v,w,Wx,Wy,Wz,Radius,Group_ID,Viscosity,Viscosity_t = rd.readVortexRingInstance(filename)

    vtrInstance = VortexRingInstance(x,y,z,u,v,w,Wx,Wy,Wz,Radius,Group_ID,Viscosity,Viscosity_t)

    times.append(stamp*time_step_size)
    enstrophies.append(calcEnstrophy_vec(vtrInstance))
    print(calcEnstrophy_vec(vtrInstance))

plt.plot(times, enstrophies)
plt.show()