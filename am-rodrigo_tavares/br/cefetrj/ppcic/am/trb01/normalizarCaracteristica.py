"""
PPCIC - Aprendizado de Máquina - Prof. Eduardo Bezerra
Aluno: Rodrigo Tavares de Souza
"""

import numpy as np

def normalizarCaracteristica(x):

    if len(x.shape) == 1:
        n = 1
    else:
        n = x.shape[1]

    nx = np.zeros(x.shape)
    u = np.zeros(n)
    s = np.zeros(n)

    for i in range(0, n):

        if n > 1:
            u[i] = np.mean(x[:,i])
            s[i] = np.std(x[:,i])
            nx[:,i] = (x[:,i] - u[i]) / s[i]

        else:
            u = np.mean(x)
            s = np.std(x)
            nx = (x - u) / s

    return nx, u, s