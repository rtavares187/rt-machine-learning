"""
PPCIC - Aprendizado de Máquina - Prof. Eduardo Bezerra
Aluno: Rodrigo Tavares de Souza
"""

import numpy as np
from sigmoide import *
#import warnings
# ignorando warning pois quando h = 1 executa log de zero e o numpy trata internamente convertendo pra nan
#warnings.filterwarnings("ignore")

def computarCustoRL(theta, x, y):

    m = len(x)

    h = sigmoide(np.dot(x, theta))

    # tratando os valores utilizados no log no cálculo do custo para não fazer log de zero
    hm = 1 - h
    hm[hm == 0] = np.inf
    h[h == 0] = np.inf

    custo = (-1 / m) * ((np.log(h).T.dot(y)) + (np.log(hm).T.dot(1 - y)))

    return custo