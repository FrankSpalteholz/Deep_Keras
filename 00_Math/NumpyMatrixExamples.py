import numpy as np
from sympy import *
from numpy.linalg import inv


# Numpy stands for "numeric Python" while Sympy stands for "symbolic Python".
#
# In Numpy, every calculation is numeric. Everything is made for very fast computing.
# That's why, for example, Numpy matrices must have a type (as in other languages) a
# nd the computations may vary from a type to another.
# They are more useful for engineers using very large matrices and making many computations.
#
# On the other hand, everything in Sympy is symbolic. Computing is made using CAS
# (Computer Algebra System) and results are exact when possible.
# Sympy matrices don't need to be typed and may include symbolic variables.
# They are more useful for students learning algebra.


PI = 3.14159265359
degree = 30

rad = (degree*PI)/180

rad = np.radians(30)

thetaX = rad
thetaY = rad
thetaZ = rad

MRotX = np.array([[1, 0, 0],
                  [0, np.cos(thetaX), -np.sin(thetaX)],
                  [0,np.sin(thetaX), np.cos(thetaX)]])

MRotY = np.array([[np.cos(thetaY), 0, np.sin(thetaY)],
                  [0, 1, 0],
                  [-np.sin(thetaY), 0, np.cos(thetaY)]])

MRotZ = np.array([[np.cos(thetaZ), -np.sin(thetaZ), 0],
                  [np.sin(thetaZ), np.cos(thetaZ), 0],
                  [0, 0, 1]])

MRot = np.matmul(np.matmul(MRotX,MRotY),MRotZ)

print("Rot Matrices ///////////////////////////////////////////////////////")

print(f'MatrixX:\n{MRotX}\n')
print(f'MatrixY:\n{MRotY}\n')
print(f'MatrixZ:\n{MRotZ}\n')

print("Transpose and Inverse //////////////////////////////////////////////")

print(f'Original Matrix:\n{MRot}\n')

MRot_trans = MRot.transpose()
print(f'Transposed Matrix:\n{MRot_trans}\n')

MRot_inv = inv(MRot)
print(f'Inverse Matrix:\n{MRot_inv}\n')

print("Det ///////////////////////////////////////////////////////////////")

MRot_det = np.linalg.det(MRot)
print(f'Matrix det:\n{MRot_det}\n')

print("Eig ///////////////////////////////////////////////////////////////")

MRot_eig = np.linalg.eig(MRot)
print("Eigenvalues: {}".format(len(MRot_eig[0])))
print("Eigenvectors: {}".format(len(MRot_eig[1])))

print(f'Matrix eig-vals:')
for eigVal in MRot_eig[0]:
    print(format(eigVal))
print()
print(f'Matrix eig-vecs:')
for eigVec in MRot_eig[1]:
    print(format(eigVec))
print()
print("Rref ///////////////////////////////////////////////////////////////")

# sympy rref version
M = Matrix([[1, 0, 1, 3], [2, 3, 4, 7], [-1, -3, -3, -4]])
print("Sympy Matrix: \n{} ".format(M))

# Use sympy.rref() method
M_rref = M.rref()

print("Rref Sympy: {}\n".format(M_rref))

print("QR /////////////////////////////////////////////////////////////////")

# scipy rref version
M2 = np.array([[1, 0, 1, 3], [2, 3, 4, 7], [-1, -3, -3, -4]])
print("Scipy Matrix: \n{} ".format(M2))

(M2_, M2_rref) = np.linalg.qr(M2)

print(f'Q Numpy:\n{M2_rref}')
print(f'R Numpy:\n{M2_}\n')

print("SVD ///////////////////////////////////////////////////////////////")

(U, S, V) = np.linalg.svd(M2)

print(f'U:\n{U}')
print(f'S:\n{S}\n')
print(f'V:\n{V}\n')











































