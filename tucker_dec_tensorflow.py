import tensorly as tl
# tl.set_backend('tensorflow')
import numpy as np
import os
import tensorflow as tf
tf.enable_eager_execution()

# tfe = tf.contrib.eager

from tensorly.decomposition import tucker, partial_tucker
from tensorly.tucker_tensor import tucker_to_tensor
from tensorly.tenalg import mode_dot
# from tensorly.random import check_random_state
# from tensorly.metrics import RMSE

# num_epochs = 5000
# penalty = 0.001
# lr = 0.1
# random_state = 1234
# rng = check_random_state(random_state)

TENSORS_PATH = os.path.join('data', 'tensors')
# tensor = tfe.Variable(tl.tensor(np.load(os.path.join(TENSORS_PATH, 'weights.npy'))))
# rank = [1, 1, 8, 8]

tensor = np.load(os.path.join(TENSORS_PATH, 'weights.npy'))
# tucker2
# rank = [int(tensor.shape[0]), int(tensor.shape[1]), 5, 5]
# tucker1
rank = [int(tensor.shape[0]), int(tensor.shape[1]), int(tensor.shape[2]), 1]
identity_factors = [np.identity(i) for i in rank[:3]]
core, factors = partial_tucker(tensor, modes=[3], ranks=rank, n_iter_max=100)

rec = tl.tucker_to_tensor(core, identity_factors + factors)
rec_error = np.linalg.norm(rec - tensor)
print(rec_error)
# 78.71723244893371

#%% Convolution operation
G = core  # core filter
A = factors[0]  # factor matrix
X = np.random.rand(1,300,300,24)  # input

# standard convolution
std_conv = tf.nn.conv2d(X, tensor, strides=[1,1,1,1], padding="SAME")

# tucker1 convolution
tuck1_conv1 = tf.nn.conv2d(X, G, strides=[1,1,1,1], padding="SAME")
tuck1_conv = mode_dot(tuck1_conv1, A, mode=3)

conv_error = np.linalg.norm(tuck1_conv - std_conv)
print(conv_error)
# 13677.724001243612
# este error es esperado por la compresion y la division de la operacion de convolucion,
# esto se resuelve entrenando de nuevo la red con esta sustitucion de la operacion de convolucion
# este error se podria reducir previamente ajusta los parametros,
# pero se de esta forma el tensor nucleo seguiria conteniendo informacion del tensor original???

'''
#%% PCA
tensor_4 = tl.unfold(tensor, mode=3)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import stats

X_std = StandardScaler().fit_transform(tensor_4)

# covariance matrix (between each feature (each fiber mode-4) in the data)
cov_mat = np.cov(X_std.T)

# eigendecomposition
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

pca = PCA(n_components=24)
pca.fit_transform(tensor_4)
print(pca.explained_variance_ratio_)

# explained variance
pca = PCA().fit(X_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()
'''











""" tensorflow decomposition
# Variables
core = tfe.Variable(tl.tensor(rng.random_sample(rank)))
factor_ident = [tfe.Variable(tl.tensor(np.identity(i)), trainable=False) for i in rank[:2]]
factor_rnd = [tfe.Variable(tl.tensor(rng.random_sample((tensor.shape[i+2], rank[i+2])))) for i in
           range(len(rank[2:]))]
factors = factor_ident + factor_rnd

# factors = [tfe.Variable(tl.tensor(rng.random_sample((tensor.shape[i], rank[i])))) for i in
#            range(len(tensor.get_shape()._dims))]

# var = [core] + factors

# Let's define our optimiser
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

# Now we just iterate through the training loop and backpropagate
rec_error = np.Infinity
i = 0
# for epoch in range(num_epochs):
while rec_error > 51.92:
    with tfe.GradientTape() as tape:
        rec = tl.tucker_to_tensor(core, factors)
        loss_value = tf.norm(rec - tensor, ord=2)
        for f in factors:
            loss_value = loss_value + penalty * tf.norm(f, 2)

    grads = tape.gradient(loss_value, var)
    optimizer.apply_gradients(zip(grads, var),
                              global_step=tf.train.get_or_create_global_step())
    rec_error = tl.norm(rec - tensor, 2)
    if i % 100 == 0:
        print("Epoch %.3d: Loss: %.4f" % (i, rec_error))
    i += 1
"""

