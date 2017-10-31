"""
PPCIC - Aprendizado de Máquina - Prof. Eduardo Bezerra
Aluno: Rodrigo Tavares de Souza
"""

# TRABALHO 2 - PARTE A

# 1.1 Visualização dos Dados

from util import *
from graficos import *
from mapFeature import *
from costFunctionReg import *
from plotDecisionBoundary import *
from linearRegCostFunction import *
from learningCurve import *
from polyFeatures import *
from normalizarCaracteristica import *
from estimativaGaussian import *
from selectThreshold import *
from cofiCostFunc import *

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

X, y, Xval, yval, Xtest, ytest = loadMatLabData("data/ex5data1.mat")

# 2.1 Visualização dos Dados

plt = graficoDispersaoRLR(X, y)
plt.savefig("image\\fig_3_visualizacao_conjunto_treinamento.png")
plt.clf()

# 2.2 Função de custo da regressão linear regularizada

theta = np.ones(2)
gx = np.c_[np.ones(X.shape[0]),X]

_lambda = 0

cost = linearRegCostFunction(theta, gx, y, _lambda)
print("Custo com thetas iguais a um: " + str(cost))

theta = gdLinearReg(theta, gx, y, _lambda)
print("Gradientes com thetas inicializados com 1: theta[0] = " + str(theta[0]) + " e theta[1] = " + str(theta[1]))

# 2.4 Ajustando os parâmetros da regressão linear

nx = np.linspace(X.min(), X.max())
h = np.c_[np.ones(nx.shape[0]), nx].dot(theta)

plt = graficoModeloLinear(X, y, nx, h)
plt.savefig("image\\fig_4_modelo_linear.png")
plt.clf()

# 3 Viés-Variância

# 3.1 Curvas de Aprendizado

_lambda = 0.5 # valor aleatório de lambda escolhido para teste

gXval = np.c_[np.ones(Xval.shape[0]),Xval]

histJTrain, histJCV = learningCurve(gx, y, gXval, yval, _lambda)

plt = graficoCurvaAprendizado(histJTrain, histJCV)
plt.savefig("image\\fig_5_curva_aprendizado.png")
plt.clf()

# 4 Regressão Polinomial

grau = 8

xnpoli = polyFeatures(X, grau)
xvalnpoli = polyFeatures(Xval, grau)

xnpoli, u, s = normalizarCaracteristica(xnpoli)
xvalnpoli, uu, ss = normalizarCaracteristica(xvalnpoli)

xnpoli = np.c_[np.ones(xnpoli.shape[0]),xnpoli]
xvalnpoli = np.c_[np.ones(xvalnpoli.shape[0]),xvalnpoli]

# 5 Regressão Polinomial - aprendizado

_lambda = 0

theta = np.zeros(len(xnpoli[0]))

theta = gdLinearReg(theta, xnpoli, y, _lambda)
print("Valores ótimos para theta: " + getStrTheta(theta))

ohx = np.arange(X.min()-10, X.max()+10)
hx = np.copy(ohx)
hx = polyFeatures(hx, grau)
hx, uhx, shx = normalizarCaracteristica(hx)
hx = np.c_[np.ones(hx.shape[0]), hx]

hy = hx.dot(theta)

plt = graficoAjustePolinomial(X, y , ohx, hy)
plt.savefig("image\\fig_6_ajuste_polinomial.png")
plt.clf()

histJTrain, histJCV = learningCurve(xnpoli, y, xvalnpoli, yval, _lambda)

plt = graficoCurvaAprendizado(histJTrain, histJCV)
plt.savefig("image\\fig_7_curva_aprendizado_polinomial.png")
plt.clf()

# OBS: Valores dos custos no conjunto de treinamento parecem uma reta no gráfico, pois são consideravelmente
# menores do que os custos do conjunto de validação. Observamos que o theta com 5 exemplos no CT reflete em um menor
# custo no CV.

# 6 Tarefas adicionais (OPCIONAIS)

# 1)

# O enunciado pede valores de lambda de 1 a 100, mas o gráfico está de 0 a 10.
# Então testei de 1 a 100 e utilizei de 0 a 10 para a geração do gráfico.

thetaZero = np.zeros(len(xnpoli[0]))

lambdaVec = np.zeros(11)
jvec = np.zeros(11)
jvecCV = np.zeros(11)

for i in range(0, 11):

    _lambda = i
    lambdaVec[i] = _lambda

    theta = gdLinearReg(thetaZero, xnpoli, y, _lambda)

    j = linearRegCostFunction(theta, xnpoli, y, 0)
    jvec[i] = j

    jcv = linearRegCostFunction(theta, xvalnpoli, yval, 0)
    jvecCV[i] = jcv

plt = graficoTesteLambda(lambdaVec, jvec, jvecCV)
plt.savefig("image\\fig_8_curva_aprendizado_ajuste_polinomial.png")
plt.clf()

# 2) λ no intervalo {0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10}

lambdaVec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
jvec = np.zeros(10)
jvecCV = np.zeros(10)

for i in range(0, 10):

    _lambda = lambdaVec[i]

    theta = gdLinearReg(thetaZero, xnpoli, y, _lambda)

    j = linearRegCostFunction(theta, xnpoli, y, 0)
    jvec[i] = j

    jcv = linearRegCostFunction(theta, xvalnpoli, yval, 0)
    jvecCV[i] = jcv

plt = graficoTesteLambda(lambdaVec, jvec, jvecCV)
plt.savefig("image\\fig_9_curva_aprendizado_ajuste_polinomial.png")
plt.clf()

# Como podemos observar no gráfico o melhor lambda em torno de 3

# TRABALHO 2 - PARTE B

# 2.1 Distribuição Gaussiana

X, Xval, yval = loadMatLabDataB("data/ex8data1.mat")

# 2.2 Estimativa de parâmetros para uma gaussiana

mu, sigma2 = estimativaGaussian(X)

plt = graficoDistribuicaoGaussiana(X[:,0], X[:,1])
plt.savefig("image\\fig_10_distribuicao_gaussiana.png")
plt.clf()

# 2.3 Selecionando e

p = aplicacaoModelo(Xval, mu, sigma2)

e, f1 = selectThreshold(p, yval)

print("Valor de epsilon: " + str(e))
print("Valor de f1: " + str(f1))

# 3 Sistemas de Recomendação

# 3.1 Conjunto de dados de classificações de filme

Y, R = loadMatLabDataC('data/ex8_movies.mat')

X, Theta = loadMatLabDataD('data/ex8_movieParams.mat')

Xlim = X[:,:100]
Thetalim = Theta[:,:100]

# Pediu pra pegar 100 colunas de X e Theta, mas as matrizes são nm x 10 e nu x 10

# 3.2 Algoritmo de aprendizagem de filtragem colaborativa

# 3.2.1 Função de custo da filtragem colaborativa

custo, grad = cofiCostFunc(Xlim, Thetalim, Y, R)
print("Custo = " + str(custo))

# Custo diferente do enunciado quando calculado para todos os filmes, todos os usuários e todas as características

# 3.2.2 Gradiente de filtragem colaborativa
print("Gradiente = " + str(grad))
