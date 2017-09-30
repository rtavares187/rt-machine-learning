"""
PPCIC - Aprendizado de Máquina - Prof. Eduardo Bezerra
Aluno: Rodrigo Tavares de Souza
"""

import numpy as np
from computarCusto import *

difIter = 0.0001
maxIter = 4000

def gduni(alpha, gx, y, theta):

    m = len(gx)
    histJ = np.zeros(maxIter)

    numIter = 0
    dif = 999

    while not (checkConvergence(dif, numIter)):

        h = gx.dot(theta)

        # Usando X transposto pra ficar com a primeira linha com valores iguais a 1
        # para multiplicar pelo theta zero que é o termo independente da hipótese
        thetaUpdate = alpha * (1 / m) * (gx.T.dot(h - y))

        theta = theta - thetaUpdate

        j = computarCusto(gx, y, theta)
        histJ[numIter] = j

        lastJ = 0
        if(numIter > 0):
            lastJ = histJ[numIter - 1]

        dif = j - lastJ
        numIter = numIter + 1

    return theta, histJ

def checkConvergence(dif, numIter):

    if abs(dif) < difIter or numIter > maxIter:
        return True

    return False

def plotGradient(x, y, hy):

    plt.scatter(x, y, s=16, marker='x', label='Training Data')
    plt.plot(x, hy, label='Linear regression', c='red')

    plt.xlim(xmin=4, xmax=24)
    plt.xlabel('População (mil)')
    plt.ylabel('Lucro ($ - mil)')

    plt.legend(loc=4);

    return plt