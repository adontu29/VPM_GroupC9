import numpy as np
import ReadData as rd
import matplotlib.pyplot as plt


def compute_helicity_vectorized(x, Gamma, sigma):
    """
    Vectorized version of regularized helicity computation.

    Parameters:
        x: numpy array of shape (N, 3)
        Gamma: numpy array of shape (N, 3)
        sigma: float

    Returns:
        float - regularized helicity
    """
    N = x.shape[0]

    # Compute all pairwise differences: dx[p, q, :] = x[p] - x[q]
    dx = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # Shape (N, N, 3)

    # Cross products of all Gamma[p] x Gamma[q]
    Gamma_cross = np.cross(Gamma[:, np.newaxis, :], Gamma[np.newaxis, :, :])  # Shape (N, N, 3)

    # Dot product for each pair (p,q)
    dot = np.einsum('pqi,pqi->pq', dx, Gamma_cross)  # Shape (N, N)

    # Squared distances + sigma^2
    r2 = np.einsum('pqi,pqi->pq', dx, dx) + sigma ** 2  # Shape (N, N)
    denom = r2 ** 1.5

    # Avoid division by zero on the diagonal if needed (optional)
    np.fill_diagonal(denom, np.inf)

    # Final sum
    H_sigma = np.sum(dot / denom)
    H_sigma *= 1 / (4 * np.pi)

    return H_sigma
timeStamps = np.arange(0,1575,25)
timestep = 1
ttab=[]
htab=[]
for i in range(len(timeStamps)):
    stringtime = str(timeStamps[i]).zfill(4)
    # Debugged: Ensure correct file path format
    filename = f'dataset/Vortex_Ring_DNS_Re7500_{stringtime}.vtp'

    try:
        X, Y, Z, U, V, W, Wx, Wy, Wz, Radius, Group_ID, Viscosity, Viscosity_t = rd.readVortexRingInstance(filename)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        continue
    x = np.stack((X, Y, Z), axis=-1)
    gamma = np.stack((Wx, Wy, Wz), axis=-1)
    H = compute_helicity_vectorized(x, gamma, sigma=1.0)
    print(H)
    ttab.append(float(stringtime)*timestep)
    htab.append(H)
plt.plot(ttab, htab)
plt.ylabel('Helicity (|H|)')
plt.xlabel('Time')
plt.title('Helicity vs Time')
plt.grid(True, which="both", ls="--")
plt.show()