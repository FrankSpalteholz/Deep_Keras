import numpy as np
import matplotlib.pyplot as plt

def plotter(ax, data1, data2, param_dict):
    """
    A helper function to make a graph

    Parameters
    ----------
    ax : Axes
        The axes to draw to

    data1 : array
       The x data

    data2 : array
       The y data

    param_dict : dict
       Dictionary of kwargs to pass to ax.plot

    Returns
    -------
    out : list
        list of artists added
    """
    out = ax.plot(data1, data2, **param_dict)
    return out


# vec1 = np.array([1,-2,3])
#
# rad = np.radians(30)
#
# thetaX = rad
# MRotX = np.array([[1, 0, 0],
#                   [0, np.cos(thetaX), -np.sin(thetaX)],
#                   [0,np.sin(thetaX), np.cos(thetaX)]])
#
# print(f'MatrixX:\n{MRotX}\n')
# print(np.linalg.det(MRotX))
# print(np.matmul(np.linalg.inv(MRotX),MRotX))
# print(np.linalg.norm(vec1,ord=1))
# print(np.sqrt(1**2 + 2**2 + 3**2))
# print(np.linalg.norm(MRotX,ord='fro'))
# print(np.linalg.norm(MRotX,ord=2))

# minimize f(x) = 1/2 * L2(Ax+b)^2

A = np.array([[1, 2], [2, 1]])
x = np.array([[6], [7]])
b = np.array([[3], [5]])

print(f'A:\n{A}\n')
print(f'x:\n{x}\n')
print(f'b:\n{b}\n')

y = 0.5 * np.linalg.norm(np.matmul(A, x) - b) ** 2

ATA = np.matmul(A.T, A)

print(f'y:\n{y}\n')
print(f'ATA:\n{ATA}\n')

fig, ax = plt.subplots()  # Create a figure containing a single axes.

eta = 0.035
delta = 0.001
delta_new = 1

i = 0
list = []
while delta_new > delta:
    print('/////////////////////////////////////////\n')
    ATA = np.matmul(A.T, A)
    delta_new = np.linalg.norm(np.matmul(ATA, x) - np.matmul(A.T, b), ord='fro')
    x = x - eta * (np.matmul(ATA, x) - np.matmul(A.T, b))
    list.append(x)
    print(f'delta_new:\n{i}:{delta_new}\nx:{x}\n')
    # if i >= 1:
    #     plotter(ax, [list[i][0], list[i][1]], [list[i-1][0], list[i-1][1]], {'color': 'green',
    #                                     'linestyle': 'dashed',
    #                                     'linewidth':'1'})
    plotter(ax, list[i][0], list[i][1], {'marker': 'o',
                                         'color': (1/((i+1)*1.25), .5, 1/((i+1)*1.25)),
                                         'markersize': '1'})
    i = i + 1

plt.show()


