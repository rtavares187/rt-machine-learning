"""
PPCIC - Aprendizado de Máquina - Prof. Eduardo Bezerra
Aluno: Rodrigo Tavares de Souza
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def graficoDispersao(xAceito, yAceito, xReprovado, yReprovado):

    minx = np.minimum(xAceito.min(), xReprovado.min()) * 0.85
    maxx = np.maximum(xAceito.max(), xReprovado.max()) * 1.15

    plt.scatter(xAceito, yAceito, s=16, c='yellow', label='y = 0')
    plt.scatter(xReprovado, yReprovado, s=16, marker='+', c='black', label='y = 1')
    plt.xlim(xmin=minx, xmax=maxx)
    plt.xlabel('Teste 1')
    plt.ylabel('Teste 2')

    plt.legend(loc=4);

    return plt