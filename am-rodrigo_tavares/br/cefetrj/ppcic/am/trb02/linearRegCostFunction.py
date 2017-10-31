"""
PPCIC - Aprendizado de Máquina - Prof. Eduardo Bezerra
Aluno: Rodrigo Tavares de Souza
"""

import numpy as np
from scipy.optimize import minimize

def linearRegCostFunction(theta, x, y, _lambda):

    m = len(x)

    h = np.dot(x, theta)

    custo = (1 / (2 * m)) * np.sum(np.square(h - y)) + (_lambda / (2 * m)) * np.sum(np.square(theta[1:]))

    return custo

def gdLinearReg(theta, gx, y, _lambda):

    res = minimize(linearRegCostFunction, theta, args=(gx, y, _lambda), method=None, jac=gdLinearRegUpdate, options={'maxiter': 4000})
    return res.x

def gdLinearRegUpdate(theta, gx, y, _lambda):

    m = len(gx)

    h = gx.dot(theta)
    err = h - y

    thetaReg = np.copy(theta)
    thetaReg[0] = 0

    thetaUpdate = (1 / m) * (gx.T.dot(err)) + ((_lambda / m) * thetaReg)

    return thetaUpdate