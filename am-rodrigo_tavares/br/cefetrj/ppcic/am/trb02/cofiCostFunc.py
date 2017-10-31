"""
PPCIC - Aprendizado de Máquina - Prof. Eduardo Bezerra
Aluno: Rodrigo Tavares de Souza
"""

import numpy as np

def cofiCostFunc(X, Theta, Y, R):

    xTheta = np.concatenate((np.ravel(X), np.ravel(Theta)))

    nm = Y.shape[0]
    nu = Y.shape[1]
    nx = X.shape[1]

    X = np.matrix(np.reshape(xTheta[0:nm * nx], (nm, nx)))
    Theta = np.matrix(np.reshape(xTheta[nm * nx:], (nu, nx)))

    err = np.multiply((X * Theta.T) - Y, R)
    J = (1 / 2) * np.sum(np.power(err, 2))

    X_grad = (err * Theta)
    Theta_grad = (err.T * X)

    grad = np.concatenate((np.ravel(X_grad), np.ravel(Theta_grad)))

    return J, grad

