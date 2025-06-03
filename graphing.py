from matplotlib import pyplot as plt
from matplotlib import axis as ax
import numpy as np
import pandas as pd

def getGraph(time, data, label, ylimit,colour='r'):
    plt.scatter(time, data, edgecolors='k', color=colour, label=label)
    plt.plot(time, data, 'k-')
    plt.ylim([0, ylimit])
    plt.legend()


data1= pd.read_csv('CSV-GROUPC9-enstrophy-stuff/Vorticity 6080.csv', sep=';', decimal=',').to_numpy()

getGraph(data1[:,0], data1[:,1], label='Vorticity', ylimit=30, colour='r')

plt.ylabel(r'$\omega^*$')
plt.xlabel(r'$T$')


plt.show()