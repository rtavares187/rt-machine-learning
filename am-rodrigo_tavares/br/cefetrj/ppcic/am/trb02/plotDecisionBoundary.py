"""
PPCIC - Aprendizado de Máquina - Prof. Eduardo Bezerra
Aluno: Rodrigo Tavares de Souza
"""

import numpy as np
import matplotlib.pyplot as plt
from mapFeature import *
from graficos import *
from costFunctionReg import *

def plotDecisionBoundary(data, xAceito, yAceito, xReprovado, yReprovado, theta):

    # Gerar todos os pontos entre o x1 e x2 mínimos e máximos

    x1Interval = np.linspace(data[:, 0].min(), data[:, 0].max())
    x2Interval = np.linspace(data[:, 1].min(), data[:, 1].max())

    # Montar combinações entre os pontos

    xCombinations = np.zeros((len(x1Interval) * len(x2Interval), 2))

    pos = -1

    for i in range(0, len(x1Interval)):
        for j in range(0, len(x2Interval)):
            pos += 1
            xCombinations[pos][0] = x1Interval[i]
            xCombinations[pos][1] = x2Interval[j]

    fXCombinations = mapFeature(xCombinations)

    x1C, x2C = np.meshgrid(x1Interval, x2Interval)

    h = sigmoide(fXCombinations.dot(theta))
    h = h.reshape(x1C.shape)

    plt = graficoDispersao(xAceito, yAceito, xReprovado, yReprovado)

    plt.contour(x1C, x2C, h.T, [0.5], colors='g', linewidths=1)

    return plt



