import numpy as np
from numpy import linalg
import os
from datetime import datetime
# import tensorflow as tf
import tensorly as tl
# from tensorflow.python.client import timeline
from tensorly.decomposition import tucker, partial_tucker
from tensorly.tucker_tensor import tucker_to_tensor
from tensorly.tenalg import mode_dot

TENSORS_PATH = os.path.join('data', 'tensors')

# tensor = np.load(os.path.join(TENSORS_PATH, 'weights.npy'))
# tensor.shape
# (1, 1, 24, 64)

# tensor = np.arange(1,28).reshape(3,3,3) # (slices, rows, columns)
# print(tensor.shape)

# identity_matrix = np.identity(tensor.shape[2])

# si multiplico un tensor en un modo determinado con una matriz identidad, el resultado
# es el mismo tensor
# x = mode_dot(tensor, identity_matrix, 2)

# np.linalg.norm(tensor - x)
# 0.0
# x es igual a tensor

# slices
# tensor[0,:,:]       tensor[1,:,:]       tensor[2,:,:]
# [[1, 2, 3],         [[10, 11, 12],      [[19, 20, 21],
#  [4, 5, 6],          [13, 14, 15],       [22, 23, 24],
#  [7, 8, 9]]          [16, 17, 18]]       [25, 26, 27]]

# rows
# tensor[:,0,:]       tensor[:,1,:]       tensor[:,2,:]
# [[ 1,  2,  3],      [[ 4,  5,  6],      [[ 7,  8,  9],
#  [10, 11, 12],       [13, 14, 15],       [16, 17, 18],
#  [19, 20, 21]]       [22, 23, 24]]       [25, 26, 27]]

# columns
# tensor[:,:,0]       tensor[:,:,1]       tensor[:,:,2]
# [[ 1,  4,  7],      [[ 2,  5,  8],      [[ 3,  6,  9],
#  [10, 13, 16],       [11, 14, 17],       [12, 15, 18],
#  [19, 22, 25]]       [20, 23, 26]]       [21, 24, 27]]

# fibers (mode-1) tubes
# tensor[:,0,0]   tensor[:,0,1]   tensor[:,0,2]   tensor[:,1,0]   tensor[:,1,1]   tensor[:,1,2]   tensor[:,2,0]   tensor[:,2,1]   tensor[:,2,2]
# [ 1, 10, 19]    [ 2, 11, 20]    [ 3, 12, 21]    [ 4, 13, 22]    [ 5, 14, 23]    [ 6, 15, 24]

# fibers (mode-2) columns
# tensor[0,:,0]   tensor[0,:,1]   tensor[0,:,2]   tensor[1,:,0]   tensor[1,:,1]   tensor[1,:,2]   tensor[2,:,0]   tensor[2,:,1]   tensor[2,:,2]
# [1, 4, 7]       [2, 5, 8]       [3, 6, 9]       [10, 13, 16]    [11, 14, 17]    [12, 15, 18]

# fibers (mode-3) rows
# tensor[0,0,:]   tensor[0,1,:]   tensor[0,2,:]   tensor[1,0,:]   tensor[1,1,:]   tensor[1,2,:]   tensor[2,0,:]   tensor[2,1,:]   tensor[2,2,:]
# [1, 2, 3]       [4, 5, 6]       [7, 8, 9]       [10, 11, 12]    [13, 14, 15]    [16, 17, 18]


# matrix_mode1 = tl.unfold(tensor, mode=0) # los tubos (mode-1) son ordenados como columnas
# [[ 1,  2,  3,  4,  5,  6,  7,  8,  9],
#  [10, 11, 12, 13, 14, 15, 16, 17, 18],
#  [19, 20, 21, 22, 23, 24, 25, 26, 27]]

# matrix_mode2 = tl.unfold(tensor, mode=1) # las columnas (mode-2) son ordenadas como columnas
# [[ 1,  2,  3, 10, 11, 12, 19, 20, 21],
#  [ 4,  5,  6, 13, 14, 15, 22, 23, 24],
#  [ 7,  8,  9, 16, 17, 18, 25, 26, 27]]

# matrix_mode3 = tl.unfold(tensor, mode=2) # las filas (mode-3) son ordenadas como columnas
# [[ 1,  4,  7, 10, 13, 16, 19, 22, 25],
#  [ 2,  5,  8, 11, 14, 17, 20, 23, 26],
#  [ 3,  6,  9, 12, 15, 18, 21, 24, 27]]

# U = np.arange(1,10).reshape(3,3)

# prod_mode1 = mode_dot(tensor, U, mode=0)
# [[[ 78,  84,  90],
#   [ 96, 102, 108],
#   [114, 120, 126]],
#
#  [[168, 183, 198],
#   [213, 228, 243],
#   [258, 273, 288]],
#
#  [[258, 282, 306],
#   [330, 354, 378],
#   [402, 426, 450]]]

# np.dot(U, matrix_mode1)
# [[ 78,  84,  90,  96, 102, 108, 114, 120, 126],
#  [168, 183, 198, 213, 228, 243, 258, 273, 288],
#  [258, 282, 306, 330, 354, 378, 402, 426, 450]]

# print(prod_mode1[0,1,2])
# 108

# sum(tensor[:,1,2]*U[0,:])
# 108

# rank_mode1 = linalg.matrix_rank(matrix_mode1)


#%%###################################################################################
# Trabajando con tensores de la red convolucional

tensor = np.load(os.path.join(TENSORS_PATH, 'weights.npy'))
print(tensor.shape)
# (1, 1, 24, 64)  # (rows, columns, channels, num of filters)

matrix_mode3 = tl.unfold(tensor, mode=2)
matrix_mode4 = tl.unfold(tensor, mode=3)

# rank_mode3 = linalg.matrix_rank(matrix_mode3)
# rank_mode4 = linalg.matrix_rank(matrix_mode4)

rank_mode3 = 12
rank_mode4 = 12

core, factors = partial_tucker(tensor, [2,3], [rank_mode3, rank_mode4])

num_elem_tensor = tensor.size
num_elem_tucker = core.size + factors[0].size + factors[1].size
print(num_elem_tensor)
print(num_elem_tucker)

print(num_elem_tucker/num_elem_tensor)

factors_matrix = []
factors_matrix.append(np.array([1]).reshape(1,1))
factors_matrix.append(np.array([1]).reshape(1,1))
factors_matrix.append(factors[0])
factors_matrix.append(factors[1])

rec = tucker_to_tensor(core, factors_matrix)

error = np.linalg.norm(tensor - rec)
print(error)
