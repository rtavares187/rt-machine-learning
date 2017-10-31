import numpy as np
import matplotlib.pyplot as plt

a = np.array([2, 3, 4])
b = np.array([5, 6, 7])

print (np.multiply(a, b))

t = np.array([-3.63029144, 1.16636235])
r = np.array([1, 70000]).dot(t)
print(r)

a = np.array([0, 0, 0])
print (a)
a = a.reshape(-1,1)
print (a)

ar = np.array([1,3, 5, 7, 8, 9, 10])
print (ar.min())

print (np.logspace(-2, 3, 20))

x = np.array([1, 2, 3])
y = np.array([10, 20, 30])
XX, YY = np.meshgrid(x, y)

ZZ = XX + YY

print (ZZ)

#theta = np.r_[thetaZero[None,:],thetaUm[None,:]].T