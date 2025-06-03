from matplotlib import pyplot as plt
from matplotlib import axis as ax
import numpy as np


def getGraph(time, data, label, ylimit, xlabel, ylabel, colour='r'):
    plt.scatter(time, data, edgecolors='k', color=colour, label=label)
    plt.plot(time, data, 'k-')
    plt.ylim([0, ylimit])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()


def getDoubleGraph(time1,time2,data1,data2,label1,label2,ylimit,color1,color2,xlabel,ylabel):
    plt.scatter(time1, data1, edgecolors='k', color=color1, label=label1)
    plt.plot(time1, data1, 'k-')
    plt.scatter(time2, data2, edgecolors='k', color=color2, label=label2)
    plt.plot(time2, data2, 'k-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim([0, ylimit])
    plt.legend()

