"""
PPCIC - Aprendizado de Máquina - Prof. Eduardo Bezerra
Aluno: Rodrigo Tavares de Souza
"""

import numpy as np
from sigmoide import *
from funcaoCustoRegressaoLogistica import *
from scipy.optimize import minimize

maxIter = 4000
eps = 0.0001

def gdRL(theta, gx, y, alpha):

    m = len(gx)
    histJ = np.zeros(maxIter)

    numIter = 0
    dif = None

    while not (checkConvergence(dif, numIter)):

        h = sigmoide(gx.dot(theta))
        err = h - y

        thetaUpdate = alpha * (1 / m) * (gx.T.dot(err))

        oldTheta = theta
        theta = theta - thetaUpdate

        histJ[numIter] = computarCustoRL(theta, gx, y)

        dif = abs(np.linalg.norm(theta) - np.linalg.norm(oldTheta))
        numIter = numIter + 1

    return theta, histJ, numIter

def checkConvergence(dif, numIter):

    if dif is None:
        return False

    if (dif < eps) or numIter >= maxIter:
         return True

    return False

def gdRLOpt(theta, gx, y):

    res = minimize(computarCustoRL, theta, args=(gx, y), method=None, jac=gdRLOptUpdate, options={'maxiter': 4000})
    return res.x

def gdRLOptUpdate(theta, gx, y):

    m = len(gx)

    h = sigmoide(gx.dot(theta))
    err = h - y

    thetaUpdate = (1 / m) * (gx.T.dot(err))

    return thetaUpdate

def predizer(x, theta, lim):

    percent = sigmoide(x.dot(theta))

    percent = (percent >= lim).astype('int')

    return percent

