import numpy as np
import matplotlib.pyplot as plt
import ReadData2 as rd
import math as m

# === Helper Function ===
def calc_dist(p1, p2):
    return np.linalg.norm(p1 - p2)

# === Time Setup ===
time_stamps = np.arange(0, 1575, 25)  # in milliseconds
n_frames = len(time_stamps)

# === Initialize Arrays ===
velocity = np.zeros(n_frames)
ring_radius = np.zeros(n_frames)
nu = np.zeros(n_frames)
saffman_velocity = np.zeros(n_frames)
gamma = np.zeros(n_frames)
ring_pos = []

# === Loop Over Time Frames ===
for i, t in enumerate(time_stamps):
    filename = f"dataset/Vortex_Ring_DNS_Re7500_{str(t).zfill(4)}.vtp"

    try:
        X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance(filename)
    except FileNotFoundError:
        print(f"Warning: File {filename} not found. Skipping.")
        continue

    # Get ring position and radius
    ring_radius[i], ring_center = rd.getRingPosRadius(X, Y, Z, Wx, Wy, Wz)
    ring_pos.append(np.array(ring_center))

    # Viscosity
    nu[i] = Viscosity[0] if np.ndim(Viscosity) == 1 else Viscosity[1][0]

    # Circulation strength (gamma)
    strength_mag = np.sqrt(Wx**2 + Wy**2 + Wz**2)
    gamma[i] = np.sum(strength_mag)

# Convert ring position list to array
ring_pos = np.array(ring_pos)

# === Compute Translational Velocity ===
for i in range(n_frames):
    if i == 0:
        velocity[i] = calc_dist(ring_pos[i+1], ring_pos[i]) / (time_stamps[i+1] - time_stamps[i]) * 1000
    elif i == n_frames - 1:
        velocity[i] = calc_dist(ring_pos[i], ring_pos[i-1]) / (time_stamps[i] - time_stamps[i-1]) * 1000
    else:
        velocity[i] = calc_dist(ring_pos[i+1], ring_pos[i-1]) / (time_stamps[i+1] - time_stamps[i-1]) * 1000

# === Compute Saffman Velocity (skip first frame) ===
eps = 1e-8
time_sec = time_stamps / 1000
for i in range(1, n_frames):
    core_size = np.sqrt(nu[i] * time_sec[i]) + eps
    saffman_velocity[i] = (gamma[i] / (4 * np.pi * ring_radius[i])) * (
        np.log(4 * ring_radius[i] / core_size) - 0.558 - (3.6716 * nu[i] * time_sec[i] / (ring_radius[i]**2))
    )

# === Plotting ===
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(time_stamps, velocity, 'b-', label='Numerical Velocity')
ax.plot(time_stamps[1:], saffman_velocity[1:], 'r--', label='Saffman Velocity')
ax.set_xlabel("Time [ms]")
ax.set_ylabel("Velocity [m/s]")
ax.set_title("Ring Translational Velocity vs. Saffman Velocity")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# === Final Values Debug Print ===
i = n_frames - 1
print(f"gamma[{i}] = {gamma[i]:.4e}")
print(f"ringRadius[{i}] = {ring_radius[i]:.4f}")
print(f"nu[{i}] = {nu[i]:.4e}")
print(f"timeStamps[{i}] = {time_stamps[i] / 1000:.4f} s")
print(f"saffmanVelocity[{i}] = {saffman_velocity[i]:.4f}")
