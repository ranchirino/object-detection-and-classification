import numpy as np
import os
from datetime import datetime
# import tensorflow as tf
# from tensorflow.python.client import timeline
from tensorly.decomposition import tucker, partial_tucker
from tensorly.tenalg import mode_dot

TENSORS_PATH = os.path.join('data', 'tensors')

tensor = np.load(os.path.join(TENSORS_PATH, 'weights.npy'))
tensor.shape
# (1, 1, 24, 64)

identity_matrix = np.identity(tensor.shape[2])

# si multiplico una tensor en un modo determinado con una matriz identidad, el resultado
# es el mismo tensor
x = mode_dot(tensor, identity_matrix, 2)

np.linalg.norm(tensor - x)
# 0.0
# x es igual a tensor

