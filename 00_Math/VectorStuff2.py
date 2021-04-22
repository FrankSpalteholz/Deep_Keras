import numpy as np

c = np.array([[0],[-1]])

w = np.array([[1],[-2]])

W = np.array([[1,1],[1,1]])

X = np.array([[0,0],[0,1],[1,0],[1,1]])

#print(X)
print(np.matmul(X,W))
print(np.matmul(X,W)+c.T)

Y = np.matmul(X,W)+c.T


print(Y[0])



print(np.matmul(Y,w))
