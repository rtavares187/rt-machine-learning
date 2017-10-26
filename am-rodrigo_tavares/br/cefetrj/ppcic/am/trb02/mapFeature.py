"""
PPCIC - Aprendizado de Máquina - Prof. Eduardo Bezerra
Aluno: Rodrigo Tavares de Souza
"""

import numpy as np

def mapFeature(x):

    qtdCols = ((len(x[0]) + 1) * 6) - 2
    px = np.c_[np.c_[np.ones(x.shape[0]), x], np.zeros((len(x), qtdCols))]

    x1 = px[:,1]
    x2 = px[:,2]
    px[:,3] = np.multiply(x1, x2)

    exp = 2

    for i in range(4, len(px[0]), 3):

        px[:, i] = np.power(x1, exp)
        px[:, i + 1] = np.power(x2, exp)
        px[:, i + 2] = np.power(np.multiply(x1, x2), exp)

        exp += 1

    return px