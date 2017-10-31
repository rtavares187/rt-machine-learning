"""
PPCIC - Aprendizado de Máquina - Prof. Eduardo Bezerra
Aluno: Rodrigo Tavares de Souza
"""

import numpy as np

def polyFeatures(x, p):

    X_poli = np.zeros((len(x), (p - 1)))

    exp = 1

    for i in range(0, (p - 1)):

        exp += 1
        X_poli[:,i] = np.power(x, exp)

    X_poli = np.c_[x, X_poli]

    return X_poli