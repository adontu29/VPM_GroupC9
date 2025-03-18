import numpy as np

#Extended Saffman velocity model:
u = Gamma/(4*np.pi*R)*(np.log(4R/np.sqrt(nu*t))-0.558-3.6716*nu*t/R**2)
