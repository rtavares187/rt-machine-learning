"""
PPCIC - Aprendizado de Máquina - Prof. Eduardo Bezerra
Aluno: Rodrigo Tavares de Souza
"""

# 1.1 Visualização dos Dados

from util import *
from graficos import *
from mapFeature import *
from costFunctionReg import *
from plotDecisionBoundary import *

data = loadData('data/ex2data2.txt')

iAceito = data[:,2] == 0
iReprov = data[:,2] == 1

xAceito = data[iAceito][:,0]
yAceito = data[iAceito][:,1]
xReprovado = data[iReprov][:,0]
yReprovado = data[iReprov][:,1]

plt = graficoDispersao(xAceito, yAceito, xReprovado, yReprovado)
plt.savefig("image\\fig_1_dispersao_dados.png")
plt.clf()

# 1.2 Mapeamento de características (feature mapping)

y = data[:, 2]
px = mapFeature(data[:, 0:2])

# 1.3 Função de custo e gradiente

theta = np.zeros(len(px[0]))

_lambda = 1

cost = costFunctionReg(theta, px, y, _lambda)
print("Custo com thetas iguais a zero: " + str(cost))

alpha = 0.01

theta = gdRLOpt(theta, px, y, _lambda)

print("Valores ótimos para theta: " + getStrTheta(theta))

# 1.4 Esboço da fronteira de decisão

plt = plotDecisionBoundary(data, xAceito, yAceito, xReprovado, yReprovado, theta)
plt.savefig("image\\fig_2_curva_contorno.png")
plt.clf()

# 2 Regressão Linear com Regularizaçao