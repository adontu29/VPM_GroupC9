from matplotlib import pyplot as plt
from matplotlib import axis as ax
import numpy as np


def getGraph(time, data, label, ylimit,colour='r'):
    plt.scatter(time, data, edgecolors='k', color=colour, label=label)
    plt.plot(time, data, 'k-')
    plt.ylim([0, ylimit])
    plt.legend()


