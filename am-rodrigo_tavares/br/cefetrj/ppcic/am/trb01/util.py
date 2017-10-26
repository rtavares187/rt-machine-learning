import numpy as np

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
