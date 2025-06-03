from matplotlib import pyplot as plt
from matplotlib import axis as ax
import numpy as np


def getGraph(time, data, label, ylimit,colour='r'):
    plt.scatter(time, data, edgecolors='k', color=colour, label=label)
    plt.plot(time, data, 'k-')
    plt.ylim([-ylimit, ylimit])
    plt.legend()
#getGraph(np.linspace(0,10),np.linspace(0,10),label='asdas', colour='b', ylimit = 8)
#plt.show()

