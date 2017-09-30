import numpy as np
import matplotlib.pyplot as plt


print (np.logspace(-2, 3, 20))

x = np.array([1, 2, 3])
y = np.array([10, 20, 30])
XX, YY = np.meshgrid(x, y)

ZZ = XX + YY

print (ZZ)

#theta = np.r_[thetaZero[None,:],thetaUm[None,:]].T