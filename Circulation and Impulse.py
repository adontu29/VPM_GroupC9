import numpy as np
import matplotlib.pyplot as plt
import ReadData as rd
import math as m

def compute_circulation(Wx, Wy, Wz, Radius):
    """
    Compute total circulation from vortex particle data.
    Assumes spherical particles with volume ~ (4/3)πR³.
    """
    omega_mag = np.sqrt(Wx**2 + Wy**2 + Wz**2)

    # Handle Radius shape (1D or nested)
    if np.ndim(Radius) == 1:
        r = Radius
    else:
        r = Radius[0]

    particle_volume = (4/3) * np.pi * r**3

    # Circulation = sum(|omega| * volume)
    return np.sum(omega_mag * particle_volume)

# --- Initialization ---
timeStamps = np.arange(0, 1575, 25)
circulation = np.zeros(len(timeStamps))
impulse = np.zeros(len(timeStamps))
ringRadius = np.zeros(len(timeStamps))

# We'll pick R0 from first valid timestep
R0_const = None

# --- Main loop ---
for i, t in enumerate(timeStamps):
    stringtime = str(t).zfill(4)
    filename = f'dataset/Vortex_Ring_DNS_Re7500_{stringtime}.vtp'

    try:
        X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance(filename)
    except FileNotFoundError:
        print(f"File {filename} not found.")
        continue

    # Circulation from vorticity and particle volume
    Gamma = compute_circulation(Wx, Wy, Wz, Radius)
    circulation[i] = Gamma

    # Get ring radius using your ring detection method
    R, ringPos = rd.getRingPosRadius(X, Y, Z, Wx, Wy, Wz)
    ringRadius[i] = R

    # Store constant R0 if not set yet
    if R0_const is None and R > 0:
        R0_const = R

    # Use constant R0 to compute impulse
    if R0_const:
        impulse[i] = np.pi * Gamma * R0_const**2

# --- Plotting ---
plt.figure(figsize=(8, 5))
plt.plot(timeStamps / 1000, circulation, label='Circulation $\\Gamma$')
plt.xlabel('Time (s)')
plt.ylabel('Circulation')
plt.title('Circulation from Vorticity')
plt.grid(True)
plt.legend()

plt.figure(figsize=(8, 5))
plt.plot(timeStamps / 1000, impulse, label='Impulse $I = \\pi \\Gamma R_0^2$', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Impulse')
plt.title('Impulse with Constant $R_0$')
plt.grid(True)
plt.legend()

plt.show()

# --- Final printout ---
print(f"Assumed constant ring radius R0 = {R0_const:.5f}")
print(f"Final circulation Γ = {circulation[-1]:.5f}")
print(f"Final impulse I = {impulse[-1]:.5f}")
