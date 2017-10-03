"""
PPCIC - Aprendizado de Máquina - Prof. Eduardo Bezerra
Aluno: Rodrigo Tavares de Souza
"""

from gduni import *
from graficos import *
from normalizarCaracteristica import *
from gdRegressaoLogistica import *

# 1 Regressão Linear com uma Variável

# 1.1 Visualização dos Dados

data = loadData('data/ex1data1.txt')

x = data[:, 0]
y = data[:, 1]
m = len(x)

plt = graficoDispersao(x, y)
plt.savefig("image\\fig_1_plotagem_dados.png")
plt.clf()

# 1.2 Gradiente Descendente

theta = np.zeros(2)

# Criando uma primeira coluna com valores 1 para multiplicação pelo theta zero que é o termo independente da hipótese
jx = np.c_[np.ones(x.shape[0]),x]

j = computarCusto(jx, y, theta)
print("J com parâmetros(thetas) iguais a zero: " + str(j))

alpha = 0.01

theta, histJ, numIter, histTheta = gduni(alpha, jx, y, theta)
hy = jx.dot(theta.T)

hs = 'Hipótese gerada pela função gradiente: ' + str(theta[0])
hs += ' + (' + str(theta[1]) + '.x1)'
print (hs)

plt = graficoFuncaoGradiente(x, y, hy)
plt.savefig("image\\fig_2_plotagem_dados_funcao_gradiente.png")
plt.clf()

# predizer o lucro em regiões com populações de 35.000 e 70.000 habitantes

rx = np.ones((2,2))
rx[0][1] = 35000.
rx[1][1] = 70000.
ry = rx.dot(theta)

print("Previsão de lucro para população de 35.000: " + str(ry[0]) + " ($ - mil)")
print("Previsão de lucro para população de 70.000: " + str(ry[1]) + " ($ - mil)")

# 1.3 Visualização de J()

# OBS: O trecho abaixo está comentado pois gera o gráfico com o histórico de thetas e custos do
# exercício anterior. Como foram geradas 2631 iterações, a quantidade de pontos é muito grande e
# faz com que sejam calculados os custos em todas as combinações de pontos possíveis, gerando uma
# matriz J muito grande.
# Custo sobre uma grade bidimensional de valores de theta0 e de theta1

"""
thetaZero = histTheta[0:numIter,0]
thetaUm = histTheta[0:numIter,1]

# Montando a matriz de valores possíveis de theta 0 e theta 1

jC= np.zeros((len(thetaZero), len(thetaUm)))

for i in range (0, len(jC)):
    for j in range (0, len(jC[i])):
        jC[i][j] = computarCusto(jx,y, np.array([thetaZero[i], thetaUm[j]]))

thetaZeroC, thetaUmC = np.meshgrid(thetaZero, thetaUm)

plt = graficoSuperficieTxJ(thetaZeroC, thetaUmC, jC.T, theta, histTheta[0:numIter], histJ[0:numIter])
plt.savefig("image\\fig_3_superficie_theta_x_j.png")
plt.clf()
"""

# -10 <= theta0 e +10 e -1 <= theta1 <= +4 /
# No texto o segundo intervalo está theta zero, imagino que seja o theta 1

thetaZero = np.arange(-10, 10, 0.01)
thetaUm = np.arange(-1, 4, 0.01)

# Montando a matriz de valores possíveis de theta 0 e theta 1

jC= np.zeros((len(thetaZero), len(thetaUm)))

for i in range (0, len(jC)):
    for j in range (0, len(jC[i])):
        jC[i][j] = computarCusto(jx,y, np.array([thetaZero[i], thetaUm[j]]))

thetaZeroC, thetaUmC = np.meshgrid(thetaZero, thetaUm)

plt = graficoCurvaContorno(thetaZeroC, thetaUmC, jC.T, theta)
plt.savefig("image\\fig_4_curvas_contorno.png")
plt.clf()

plt = graficoSuperficie(thetaZeroC, thetaUmC, jC.T)
plt.savefig("image\\fig_5_superficie_funcao.png")
plt.clf()

# 2 Regressão Linear com Múltiplas Variáveis

# 2.1 Normalização das características

data = loadData('data/ex1data2.txt')

x1, u1, s1 = normalizarCaracteristica(np.array(data[:, 0]))
x2, u2, s2 = normalizarCaracteristica(np.array(data[:, 1]))
y = data[:, 2]

# 2.2 Gradiente descendente

gx = np.c_[np.ones(data.shape[0]),x1, x2]
theta = np.zeros(3)

theta, histJ, numIter, histTheta = gduni(alpha, gx, y, theta)
hy = gx.dot(theta.T)

hs = 'Hipótese gerada pela função gradiente multivariada: ' + str(theta[0])
hs += ' + (' + str(theta[1]) + '.x1) + (' + str(theta[2]) + '.x2)'
print (hs)

# 3 Regressão Logística

# 3.1 Visualização dos dados

data = loadData('data/ex2data1.txt')

iRep = data[:,2] == 0
iApr = data[:,2] == 1

graficoDispersao2(data[iRep][:,0], data[iRep][:,1], data[iApr][:,0], data[iApr][:,1])
plt.savefig("image\\fig_6_plotagem_dados.png")
plt.clf()

# 3.2 Implementação

# 3.2.1 Função sigmoide

sZero = sigmoide(0)
print("Função sigmoide com Z igual a zero: " + str(sZero))

# 3.2.2 Função de custo e gradiente

gx = np.c_[np.ones(data.shape[0]),np.array(data[:, 0]), np.array(data[:, 1])]
y = data[:, 2]
thetaZ = np.zeros(3)

j = computarCustoRL(thetaZ, gx, y)
print("J com parâmetros(theta) iguais a zero: " + str(j))

# 3.2.3 Aprendizado dos parâmetros

# OBS: Implementei duas versões do gradiente, uma em que controlo a atualização dos thetas "manualmente"
# e outra em que utilizo o pacote scipy.optimize.minimize. Vi na documentação que esse pacote simula a
# execução de vários alphas de forma otimizada.

# testei com vários alphas, esse foi o que minimizou mais j e se aproximou do valor 0,776 do enumciado
alpha = 0.0005

theta, histJ, numIter = gdRL(thetaZ, gx, y, alpha)

# 3.2.4 Avaliação do modelo

aval = np.array([1, 45., 85.])
print('Probabilidade de aprovação para candidato que tirou 45 e 85 nas avaliações:')

percent = sigmoide(aval.dot(theta))
print('     1) Gradiente com atualização "manual": ' + str(percent))

thetaM = gdRLOpt(thetaZ, gx, y)
percentM = sigmoide(aval.dot(thetaM))
print('     2) Gradiente utilizando scipy.optimize.minimize: ' + str(percentM))

pth = 'Thetas da Regressão Logística: theta0 = ' + str(thetaM[0])
pth += ' / theta1 = ' + str(thetaM[1]) + ' / theta2 = ' + str(thetaM[2])
print(pth)

lim = 0.5
yh = predizer(gx, thetaM, lim)

comp = y == yh
acertos = np.sum(comp.astype('int'))

percent = (acertos / len(y)) * 100

print('Porcentagem de acertos: ' + str(percent) + '%')