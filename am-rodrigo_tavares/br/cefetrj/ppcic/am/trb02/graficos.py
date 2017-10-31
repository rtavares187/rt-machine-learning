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

def graficoDispersaoRLR(x, y):

    plt.scatter(x, y, s=50, c='r', linewidths=1, marker='x')

    plt.xlabel('Mudança no nível da água (x)')
    plt.ylabel('Quantidade de água saindo (y)')

    return plt

def graficoModeloLinear(x, y, nx, ny):

    plt = graficoDispersaoRLR(x, y)

    plt.plot(nx, ny, c='blue')

    return plt

def graficoCurvaAprendizado(histJTrain, histJCV):

    plt.plot(np.arange(1, 13), histJTrain, label='Treinamento', c='blue')
    plt.plot(np.arange(1, 13), histJCV, label='Validação Cruzada', c='green')

    plt.xlabel('Número de exemplos de treinamento')
    plt.ylabel('Erro')

    plt.legend();

    return plt

def graficoAjustePolinomial(x, y , hx, hy):

    plt = graficoDispersaoRLR(x, y)
    plt.plot(hx, hy, c='blue')

    return plt

def graficoDistribuicaoGaussiana(x, y):

    plt.scatter(x, y, s=12, c='blue', linewidths=1, marker='x')

    plt.xlabel('Latência')
    plt.ylabel('Taxa de transferência')

    return plt

def graficoTesteLambda(lambdaVec, jvec, jvecCV):

    plt.plot(lambdaVec, jvec, label='Treinamento', c='blue')
    plt.plot(lambdaVec, jvecCV, label='Validação Cruzada', c='green')

    plt.xlabel('lambda')
    plt.ylabel('Erro')

    plt.legend();

    return plt