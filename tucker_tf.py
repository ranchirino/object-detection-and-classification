import tensorly as tl
import os

tl.set_backend('tensorflow')
import numpy as np
import tensorflow as tf

tfe = tf.contrib.eager

from tensorly.tucker_tensor import tucker_to_tensor
from tensorly.random import check_random_state
from tensorly.metrics import RMSE


num_epochs = 5000
penalty = 0.01
lr = 0.01

shape = [5, 5, 5]
rank = [1, 1, 8, 8]

random_state = 1234
rng = check_random_state(random_state)


# tensor = tfe.Variable(tl.tensor(rng.random_sample(shape)))
TENSORS_PATH = os.path.join('data', 'tensors')
tensor = tfe.Variable(tl.tensor(np.load(os.path.join(TENSORS_PATH, 'weights.npy'))))

core = tfe.Variable(tl.tensor(rng.random_sample(rank)))
factors = [tfe.Variable(tl.tensor(rng.random_sample((tensor.shape[i], rank[i])))) for i in
           range(len(tensor.get_shape()._dims))]

var = [core] + factors

optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

for epoch in range(num_epochs):

    with tfe.GradientTape() as tape:
        rec = tl.tucker_to_tensor(core, factors)
        loss_value = tf.norm(rec - tensor, ord=2)
        for f in factors:
            loss_value = loss_value + penalty * tf.norm(f, 2)

    grads = tape.gradient(loss_value, var)
    optimizer.apply_gradients(zip(grads, var),
                              global_step=tf.train.get_or_create_global_step())

    rec_error = tl.norm(rec - tensor, 2)
    if epoch % 100 == 0:
        print("Epoch {:03d}: Loss: {:.3f}".format(epoch, rec_error))

