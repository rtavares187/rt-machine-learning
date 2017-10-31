"""
PPCIC - Aprendizado de Máquina - Prof. Eduardo Bezerra
Aluno: Rodrigo Tavares de Souza
"""

import numpy as np
from scipy.optimize import minimize

def costFunctionReg(theta, x, y, _lambda):

    m = y.size
    h = sigmoide(x.dot(theta))

    cost = -1 * (1 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot(1 - y)) + (_lambda / (2 * m)) * np.sum(np.square(theta[1:]))

    return cost

def sigmoide(z):

    exp = np.dot(z, -1)
    v = np.exp(exp)
    return (1 / (1 + v))

def gdRLOpt(theta, gx, y, _lambda):

    res = minimize(costFunctionReg, theta, args=(gx, y, _lambda), method=None, jac=gdRLOptUpdate, options={'maxiter': 4000})
    return res.x

def gdRLOptUpdate(theta, gx, y, _lambda):

    m = len(gx)

    h = sigmoide(gx.dot(theta))
    err = h - y

    thetaReg = np.copy(theta)
    thetaReg[0] = 0

    thetaUpdate = (1 / m) * (gx.T.dot(err)) + ((_lambda / m) * thetaReg)

    return thetaUpdate