"""
PPCIC - Aprendizado de Máquina - Prof. Eduardo Bezerra
Aluno: Rodrigo Tavares de Souza
"""

import numpy as np
from linearRegCostFunction import *

def learningCurve(X, y, Xval, yval, _lambda):

    m = len(X)

    histJTrain = np.zeros((m, 1))
    histJCV = np.zeros((m, 1))

    initTheta = np.zeros(X.shape[1])

    for i in np.arange(m):

        theta = gdLinearReg(initTheta, X[:i+1], y[:i+1], _lambda)

        histJTrain[i] = linearRegCostFunction(theta, X[:i+1], y[:i+1], 0)
        histJCV[i] = linearRegCostFunction(theta, Xval, yval, 0)

    return histJTrain, histJCV

