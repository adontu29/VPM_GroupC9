import numpy as np
import math as m
import matplotlib.pyplot as plt
import ReadData as rd


def calcDist (instance1, instance2):
    return  m.sqrt((instance1[0] - instance2[0])**2 + (instance1[1] - instance2[1])**2 + (instance1[2] - instance2[2])**2)

def zeta(rho):
    return 15/8/np.pi/(rho**2+1)**(7/2)
def q(rho):
    return rho**3*(rho**2+5/2)/(rho**2+1)**(5/2)/4/np.pi
def reg_zeta(rho,radius):
    rho = rho/radius
    return 15/8/np.pi/(rho**2+1)**(7/2)/radius**3
def reg_q(rho,radius):
    rho=rho/radius
    return rho**3*(rho**2+5/2)/(rho**2+1)**(5/2)/4/np.pi


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


timeStampsNames = np.arange(0,max_timesteps+1,25)
timeStamps = np.arange(0, no_timesteps*time_step_size, time_step_size)

# Define the ranges for x, y, and z axes
xGrid = np.arange(-1.5, 1.5, 0.1)
yGrid = np.arange(-0.3, 4.3, 0.1)
zGrid = np.arange(-1.5, 1.5, 0.1)

# Create the 3D grid
XGrid, YGrid, ZGrid = np.meshgrid(xGrid, yGrid, zGrid, indexing='ij')  # 'ij' gives matrix indexing


for i in range(len(timeStamps)):
    stringtime = str(timeStampsNames[i]).zfill(4)
    # Debugged: Ensure correct file path format
    filename = f'dataset2/Vortex_Ring_{stringtime}.vtp'
    #print(filename)

    try:
        X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance(filename)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        continue

    grid_shape = XGrid.shape
    grid_points = np.stack((XGrid.ravel(), YGrid.ravel(), ZGrid.ravel()), axis=-1)
    particle_pos = np.stack((X, Y, Z), axis=-1)  # shape: (N_particles, 3)
    Gamma = np.stack((Wx, Wy, Wz), axis=-1)  # shape: (N_particles, 3), already circulation
    omega_grid = np.zeros_like(grid_points)

    for p in range(particle_pos.shape[0]):
        r = grid_points - particle_pos[p]
        r_mag = np.linalg.norm(r, axis = 1)

        r_mag[r_mag == 0] = 1e-12
        r_cubed = r_mag**3
        r_squared = r_mag**2
        omega_grid_contrib = ((reg_zeta(r_mag,particle_radius) - reg_q(r_mag,particle_radius)/r_cubed) * Gamma[p] +
                              np.cross((3*reg_q(r_mag,particle_radius)/r_cubed) - reg_zeta(r_mag,particle_radius),
                              np.dot(r,Gamma[p])/r_squared*r))
        omega_grid+=omega_grid_contrib



