"""
PPCIC - Aprendizado de Máquina - Prof. Eduardo Bezerra
Aluno: Rodrigo Tavares de Souza
"""

import numpy as np
import scipy.io as spio

def loadData(filePath):

    data = np.loadtxt(filePath, delimiter=',')
    return data

def getStrTheta(theta):

    strTheta = ""

    for i in range(0, len(theta)):

        if i != 0:
            strTheta += ", "

        strTheta += "theta" + str(i) + " = " + str(theta[i])


    return strTheta

def loadMatLabData(filePath):

    mat = spio.loadmat(filePath, squeeze_me=True)

    X = mat['X']
    y = mat['y']

    Xval = mat['Xval']
    yval = mat['yval']

    Xtest = mat['Xtest']
    ytest = mat['ytest']

    return X, y, Xval, yval, Xtest, ytest

def loadMatLabDataB(filePath):

    mat = spio.loadmat(filePath, squeeze_me=True)

    X = mat['X']
    Xval = mat['Xval']
    yval = mat['yval']

    return X, Xval, yval

def loadMatLabDataC(filePath):

    mat = spio.loadmat(filePath, squeeze_me=True)
    Y = mat['Y']
    R = mat['R']

    return Y, R

def loadMatLabDataD(filePath):

    mat = spio.loadmat(filePath, squeeze_me=True)
    X = mat['X']
    Theta = mat['Theta']

    return X, Theta