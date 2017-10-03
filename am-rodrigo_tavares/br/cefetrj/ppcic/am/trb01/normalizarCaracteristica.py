"""
PPCIC - Aprendizado de Máquina - Prof. Eduardo Bezerra
Aluno: Rodrigo Tavares de Souza
"""

import numpy as np

def normalizarCaracteristica(x):

    u = np.mean(x)
    s = np.std(x)

    xn = x - u
    xn = xn / s

    return xn, u, s