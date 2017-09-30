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

def graficoCurvaContorno(thetaZeroC, thetaUmC, jC, theta):

    # desenhar todas as curvas do intervalo do enunciado
    level = np.logspace(-1, 10, 50)

    plt.contour(thetaZeroC, thetaUmC, jC, level)
    plt.scatter(theta[0], theta[1])
    plt.xlabel('Theta 0')
    plt.ylabel('Theta 1')
    plt.tight_layout()

    return plt

def graficoSuperficie(thetaZeroC, thetaUmC, jC):

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(thetaZeroC, thetaUmC, jC)
    ax.set_zlabel('Custo')
    ax.set_zlim(jC.min(), jC.max())
    ax.set_xlabel('Theta 0')
    ax.set_ylabel('Theta 1')

    return plt