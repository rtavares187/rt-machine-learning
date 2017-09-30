"""
PPCIC - Aprendizado de Máquina - Prof. Eduardo Bezerra
Aluno: Rodrigo Tavares de Souza
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d, Axes3D

def scatterPlot(x, y):

    plt.scatter(x, y, s=16, marker='x')
    plt.xlim(xmin=4, xmax=24)
    plt.xlabel('População (mil)')
    plt.ylabel('Lucro ($ - mil)')

    return plt

def loadData(filePath):

    data = np.loadtxt('data/ex1data1.txt', delimiter=',')
    return data

def computarCusto(x, y, theta):

    m = len(x)

    h = np.dot(x, theta)

    cost = (1 / (2 * m)) * np.sum(np.square(h - y))

    return cost

def contourPlot(thetaZeroC, thetaUmC, jC, theta):

    # desenhar todas as curvas do intervalo do enunciado
    level = np.logspace(-1, 10, 50)

    plt.contour(thetaZeroC, thetaUmC, jC, level)
    plt.scatter(theta[0], theta[1])
    plt.xlabel('Theta 0')
    plt.ylabel('Theta 1')
    plt.tight_layout()

    return plt

def surfacePlot(thetaZeroC, thetaUmC, jC):

    ax = Axes3D(plt.figure())
    ax.plot_surface(thetaZeroC, thetaUmC, jC)
    ax.set_zlabel('Custo')
    ax.set_zlim(jC.min(), jC.max())
    ax.set_xlabel('Theta 0')
    ax.set_ylabel('Theta 1')

    return plt