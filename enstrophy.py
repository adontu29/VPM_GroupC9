import numpy as np

def calcEnstrophy(ringInstance):
    summing = 0
    for i in range(len(ringInstance.x)):
        for j in range(len(ringInstance.x)):
            if i != j:
                pos1 = np.array(ringInstance.x[i], ringInstance.y[i], ringInstance.z[i])
                pos2 = np.array(ringInstance.x[j], ringInstance.y[j], ringInstance.z[j])

                strength1 = np.array(ringInstance.wx[i], ringInstance.wy[i], ringInstance.wz[i])
                strength2 = np.array(ringInstance.wx[j], ringInstance.wy[j], ringInstance.wz[j])

                rho = np.linalg.norm(pos1 - pos2) / ringInstance.Radius

                factor1 = (5 - rho**2 * (rho**2 + 7/2)) / (rho**2 + 1)**(7/2)
                factor2 = 3 * (rho**2 * (rho**2 + 9/2) + 7/2) / (rho**2 + 1)**(9/2)

                summand = (factor1 * np.dot(strength1, strength2) + factor2 * np.dot((pos1 - pos2), strength1) * np.dot((pos1 - pos2), strength2)) / ringInstance.Radius**3
                summing = summing + summand

    summing = summing / (4 * np.pi)
        
