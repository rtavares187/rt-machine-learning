"""
PPCIC - Aprendizado de Máquina - Prof. Eduardo Bezerra
Aluno: Rodrigo Tavares de Souza
"""

from computarCusto import *
from gduni import *

# 1 Regressão Linear com uma Variável

# 1.1 Visualização dos Dados

data = loadData('data/ex1data1.txt')

x = data[:, 0]
y = data[:, 1]
m = len(x)

plt = scatterPlot(x, y)
#plt.show()
plt.clf()

# 1.2 Gradiente Descendente

theta = np.zeros(2)

# Criando uma primeira coluna com valores 1 para multiplicação pelo theta zero que é o termo independente da hipótese
jx = np.c_[np.ones(x.shape[0]),x]

j = computarCusto(jx, y, theta)
print("J com parâmetros iguais a zero: " + str(j))

alpha = 0.01

theta, histJ = gduni(alpha, jx, y, theta)
hy = jx.dot(theta.T)

plt = plotGradient(x, y, hy)
#plt.show()
plt.clf()

# predizer o lucro em regiões com populações de 35.000 e 70.000 habitantes

rx = np.ones((2,2))
rx[0][1] = float(35000)
rx[1][1] = float(70000)
ry = rx.dot(theta.T)

print("Previsão de lucro para população de 35.000: " + str(ry[0]) + " ($ - mil)")
print("Previsão de lucro para população de 70.000: " + str(ry[1]) + " ($ - mil)")

# 1.3 Visualização de J()

# -10 <= theta0 e +10 e -1 <= theta1 <= +4 /

# No texto o segundo intervalo está theta zero, imagino que seja o theta 1

""" 

Usando arange com incremento de 0,01, para o intervalo informado, gsão geradas 2000 posições pra theta zero e 
500 pra theta 1, então usei linspace para limitar o número de valores

thetaZero = np.arange(-10, 10, 0.01)
thetaUm = np.arange(-1, 4, 0.01)

# Limitando o tamanho do theta zero uma vez que o intervalo é maior e gera mais valores
thetaZero = thetaZero[0:len(thetaUm)]

"""

thetaZero = np.linspace(-10, 10, 120)
thetaUm = np.linspace(-1, 4, 120)

# Montando a matriz de valores possíveis de theta 0 e theta 1

thetaZeroC, thetaUmC = np.meshgrid(thetaZero, thetaUm)
jC= np.zeros((thetaZero.size, thetaUm.size))

for i in range (0, len(jC)):
    for j in range (0, len(jC[i])):
        jC[i][j] = computarCusto(jx,y, np.array([thetaZeroC[i, j], thetaUmC[i, j]]))

plt = contourPlot(thetaZeroC, thetaUmC, jC, theta)
#plt.show()
plt.clf()

plt = surfacePlot(thetaZeroC, thetaUmC, jC)
plt.show()
plt.clf()