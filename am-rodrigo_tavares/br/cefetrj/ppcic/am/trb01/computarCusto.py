"""
PPCIC - Aprendizado de Máquina - Prof. Eduardo Bezerra
Aluno: Rodrigo Tavares de Souza
"""

import numpy as np

def computarCusto(x, y, theta):

    m = len(x)

    h = np.dot(x, theta)

    custo = (1 / (2 * m)) * np.sum(np.square(h - y))

    return custo