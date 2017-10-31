"""
PPCIC - Aprendizado de Máquina - Prof. Eduardo Bezerra
Aluno: Rodrigo Tavares de Souza
"""

import numpy as np

def estimativaGaussian(X):

    mu = X.mean(axis=0)
    sigma2 = X.var(axis=0)

    return mu, sigma2

def aplicacaoModelo(Xcv, mu, sigma2):

    n = Xcv.shape[1]
    p = np.ones(Xcv.shape[0])

    for j in range(0, n):

        x = Xcv[:,j]

        pj = (1 / (np.sqrt(2 * np.pi) * np.sqrt(sigma2[j]))) * np.exp(-1 * (np.power(x - mu[j], 2) / (2 * sigma2[j])))

        p = np.multiply(p, pj)

    return p