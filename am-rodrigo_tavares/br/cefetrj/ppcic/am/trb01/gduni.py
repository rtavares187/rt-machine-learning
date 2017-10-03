"""
PPCIC - Aprendizado de Máquina - Prof. Eduardo Bezerra
Aluno: Rodrigo Tavares de Souza
"""

import numpy as np
from computarCusto import *

maxIter = 4000
eps = 0.0001

def gduni(alpha, gx, y, theta):

    m = len(gx)
    histJ = np.zeros(maxIter)
    histTheta = np.zeros((maxIter, len(theta)))

    numIter = 0
    dif = None

    while not (checkConvergence(dif, numIter)):

        h = gx.dot(theta)
        err = h - y

        # Usando X transposto, a primeira coluna com valores iguais a 1
        # para multiplicar pelo theta zero que é o termo independente da hipótese
        thetaUpdate = alpha * (1 / m) * (gx.T.dot(err))

        oldTheta = theta
        theta = theta - thetaUpdate

        histJ[numIter] = computarCusto(gx, y, theta)
        histTheta[numIter] = theta

        dif = abs(np.linalg.norm(theta) - np.linalg.norm(oldTheta))
        numIter = numIter + 1

    return theta, histJ, numIter, histTheta

def checkConvergence(dif, numIter):

    if dif is None:
        return False

    if (dif < eps) or numIter >= maxIter:
         return True

    return False