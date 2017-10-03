"""
PPCIC - Aprendizado de Máquina - Prof. Eduardo Bezerra
Aluno: Rodrigo Tavares de Souza
"""

import numpy as np

def sigmoide(z):

    exp = np.dot(z, -1)
    v = np.exp(exp)
    return (1 / (1 + v))