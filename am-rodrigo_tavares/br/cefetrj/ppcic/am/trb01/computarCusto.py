"""
PPCIC - Aprendizado de Máquina - Prof. Eduardo Bezerra
Aluno: Rodrigo Tavares de Souza
"""

import numpy as np

def loadData(filePath):

    data = np.loadtxt('data/ex1data1.txt', delimiter=',')
    return data

def computarCusto(x, y, theta):

    m = len(x)

    h = np.dot(x, theta)

    cost = (1 / (2 * m)) * np.sum(np.square(h - y))

    return cost