"""
PPCIC - Aprendizado de Máquina - Prof. Eduardo Bezerra
Aluno: Rodrigo Tavares de Souza
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def graficoDispersao(x, y):

    plt.scatter(x, y, s=16, marker='x')
    plt.xlim(xmin=4, xmax=24)
    plt.xlabel('População (mil)')
    plt.ylabel('Lucro ($ - mil)')

    return plt

def graficoFuncaoGradiente(x, y, hy):

    plt.scatter(x, y, s=16, marker='x', label='Training Data')
    plt.plot(x, hy, label='Linear regression', c='red')

    plt.xlim(xmin=4, xmax=24)
    plt.xlabel('População (mil)')
    plt.ylabel('Lucro ($ - mil)')

    plt.legend(loc=4);

    return plt

def graficoSuperficieTxJ(thetaZero, thetaUm, jC, theta, histTheta, histJ):

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.plot_surface(thetaZero, thetaUm, jC, alpha=0.6)
    ax.scatter(histTheta[:, 0], histTheta[:, 1], histJ[:], c='r', s=2)

    ax.set_zlabel('Custo')
    ax.set_zlim(0, jC.max())
    ax.set_xlim(thetaZero.min(), thetaZero.max())
    ax.set_ylim(thetaUm.min(), thetaUm.max())
    ax.set_xlabel('Theta 0')
    ax.set_ylabel('Theta 1')
    ax.view_init(elev=42, azim=32)

    return plt

def graficoCurvaContorno(thetaZero, thetaUm, jC, theta):

    # desenhar todas as curvas do intervalo do enunciado
    level = np.logspace(-1, 10, 50)

    plt.contour(thetaZero, thetaUm, jC, level)
    plt.scatter(theta[0], theta[1])
    plt.xlabel('Theta 0')
    plt.ylabel('Theta 1')
    plt.tight_layout()

    return plt

def graficoSuperficie(thetaZero, thetaUm, jC):

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(thetaZero, thetaUm, jC, alpha=0.6)
    ax.set_zlabel('Custo')
    ax.set_zlim(jC.min(), jC.max())
    ax.set_xlabel('Theta 0')
    ax.set_ylabel('Theta 1')

    return plt

def graficoDispersao2(xRep, yRep, xApr, yApr):

    minx = np.minimum(xRep.min(), xApr.min()) * 0.98
    maxx = np.maximum(xRep.max(), xApr.max()) * 1.02

    plt.scatter(xRep, yRep, s=16, c='yellow')
    plt.scatter(xApr, yApr, s=16, marker='+', c='black')
    plt.xlim(xmin=minx, xmax=maxx)
    plt.xlabel('Nota 1')
    plt.ylabel('Nota 2')

    return plt