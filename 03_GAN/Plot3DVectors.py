import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FuncAnimation

PI = 3.14159265359

def output_plot(vector_out_list):

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    axis = np.array([[0, 0, 0, 2.5, 0, 0], [0, 0, 0, -2.5, 0, 0],
                     [0, 0, 0, 0, 2.5, 0], [0, 0, 0, 0, -2.5, 0],
                     [0, 0, 0, 0, 0, 2.5], [0, 0, 0, 0, 0, -2.5]])
    X, Y, Z, U, V, W = zip(*axis)
    ax.quiver(X, Y, Z, U, V, W, arrow_length_ratio=0.0, color='black')

    for vector in vector_out_list:
        i = 0
        for rot_vec in vector[0]:
            axes = np.array([[vector[1][0], vector[1][1], vector[1][2],
                                        rot_vec[0][0], rot_vec[1][0], rot_vec[2][0]]])

            X2, Y2, Z2, U2, V2, W2 = zip(*axes)
            ax.quiver(X2, Y2, Z2, U2, V2, W2, arrow_length_ratio=0.1, color=vector[0][i][1][1])
            i = i + 1

    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])
    ax.set_zlim([-2.5, 2.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def create_rot_output_vec_list(input_vec, color):
    return list(zip(input_vec, [color, color, color]))

def create_rotation_matrix(rad_array):
    MRotX = np.array([[1, 0, 0],
                      [0, np.cos(rad_array[0]), np.sin(rad_array[0])],
                      [0, -np.sin(rad_array[0]), np.cos(rad_array[0])]])

    MRotY = np.array([[np.cos(rad_array[1]), 0, -np.sin(rad_array[1])],
                      [0, 1, 0],
                      [np.sin(rad_array[1]), 0, np.cos(rad_array[1])]])

    MRotZ = np.array([[np.cos(rad_array[2]), -np.sin(rad_array[2]), 0],
                      [np.sin(rad_array[2]), np.cos(rad_array[2]), 0],
                      [0, 0, 1]])

    return np.matmul(np.matmul(MRotX, MRotY), MRotZ)

def output_plot_animation(vector_out_list):

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    axis = np.array([[0, 0, 0, 2.5, 0, 0], [0, 0, 0, -2.5, 0, 0],
                     [0, 0, 0, 0, 2.5, 0], [0, 0, 0, 0, -2.5, 0],
                     [0, 0, 0, 0, 0, 2.5], [0, 0, 0, 0, 0, -2.5]])
    X, Y, Z, U, V, W = zip(*axis)
    ax.quiver(X, Y, Z, U, V, W, arrow_length_ratio=0.0, color='black')

    for vector in vector_out_list:
        i = 0
        for rot_vec in vector[0]:
            axes = np.array([[vector[1][0], vector[1][1], vector[1][2],
                                        rot_vec[0][0], rot_vec[1][0], rot_vec[2][0]]])

            X2, Y2, Z2, U2, V2, W2 = zip(*axes)
            ax.quiver(X2, Y2, Z2, U2, V2, W2, arrow_length_ratio=0.1, color=vector[0][i][1][1])
            i = i + 1

    ax.set_xlim([-2.5, 2.5])
    ax.set_ylim([-2.5, 2.5])
    ax.set_zlim([-2.5, 2.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

	#animation = FuncAnimation(fig, func=animation_frame, frames=np.arange(0, 10, 0.1), interval=10)

    plt.show()




degree = 30

rad = (degree*PI)/180
rad = np.radians(degree)

M_I = np.array([[1,0,0],[0,1,0],[0,0,1]])

MRot = create_rotation_matrix([rad,rad,rad])

MRot_out = np.matmul(MRot,M_I)

vec_rot_list = []
vec_trans = np.array([1,2,3] )
vec_color = [[1,0,0,1],[0,1,0,1],[0,0,1,1]]
vec_rot_list.append(create_rot_output_vec_list(MRot_out[0], vec_color[0]))
vec_rot_list.append(create_rot_output_vec_list(MRot_out[1], vec_color[1]))
vec_rot_list.append(create_rot_output_vec_list(MRot_out[2], vec_color[2]))

vec_rot_list2 = []
vec_trans2 = np.array([-0.5,-0.5,-0.5] )
vec_color2 = [[1,0,0,1],[0,1,0,1],[0,0,1,1]]
vec_rot_list2.append(create_rot_output_vec_list(MRot_out[0], vec_color2[0]))
vec_rot_list2.append(create_rot_output_vec_list(MRot_out[1], vec_color2[1]))
vec_rot_list2.append(create_rot_output_vec_list(MRot_out[2], vec_color2[2]))

vec_out_list = []
vec_out_list.append([vec_rot_list, vec_trans])
vec_out_list.append([vec_rot_list2, vec_trans2])


output_plot_animation(vec_out_list)

