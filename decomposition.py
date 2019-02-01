import numpy as np
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorly.decomposition import tucker, partial_tucker
from tensorly.tenalg import mode_dot

TENSORS_PATH = os.path.join('data', 'tensors')

tensor = np.load(os.path.join(TENSORS_PATH, 'weights.npy'))
tensor.shape
# (1, 1, 24, 64)

tensor.size
# 1536

#%% decomposition
# core, factors = tucker(tensor, ranks=[1, 1, 2, 5])

core, factors = partial_tucker(tensor, [2,3], [5,6])

core.shape
# (1, 1, 5, 6)
factors[0].shape
# (24, 5)
factors[1].shape
# (64, 6)
core.size + factors[0].size + factors[1].size
# 534

# reconstruction
tensor_hat = mode_dot(mode_dot(core, factors[0], 2), factors[1], 3)

loss = np.linalg.norm(tensor - tensor_hat)
print(loss)

#%% Convolution without decomposition
input = np.arange(2400).reshape((10,10,24))

# X = tf.placeholder(tf.float32, shape=(None, input.shape[0], input.shape[1], input.shape[2]))
# with tf.name_scope("convolution"):
#     conv_2d = tf.nn.conv2d(X, tensor, strides=[1,1,1,1], padding="VALID")

# with tf.Session() as sess:
    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()
    # output = sess.run(conv_2d, feed_dict={X: np.expand_dims(input, 0)}, options=run_options, run_metadata=run_metadata)

    # t0 = datetime.now()
    # output = sess.run(conv_2d, feed_dict={X: np.expand_dims(input, 0)})
    # print((datetime.now() - t0).total_seconds())
#     writer = tf.summary.FileWriter(logdir='data', graph=sess.graph)
#     writer.close()
#
#     tl = timeline.Timeline(run_metadata.step_stats)
#     trace = tl.generate_chrome_trace_format()
#     with open('timeline1.json', 'w') as f:
#         f.write(trace)


# output.shape
# (1, 10, 10, 64)

#%% Convolution with decomposition
input.shape
# (10, 10, 24)


X1 = tf.placeholder(tf.float32, shape=(input.shape[0], input.shape[1], input.shape[2]))
u1 = np.expand_dims(factors[0], 0)
u2 = np.expand_dims(np.transpose(factors[1]), 0)

with tf.name_scope("convolution"):
    mul = tf.tensordot(X1, factors[0], [[2], [0]])
    # z1 = tf.nn.conv1d(X1, u1, stride=1, padding='VALID')
    # z1_expanded = tf.expand_dims(z1, 0)
    # z2 = tf.nn.conv2d(z1_expanded, core, strides=[1,1,1,1], padding="VALID")
    # z2_contracted = tf.squeeze(z2)
    # y = tf.nn.conv1d(z2_contracted, u2, stride=1, padding='VALID')

with tf.Session() as sess:
    t1 = datetime.now()
    m = sess.run(mul, feed_dict={X1: input})
    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()
    # Y = sess.run(y, feed_dict={X1: input}, options=run_options, run_metadata=run_metadata)
    # loss = tf.norm(tf.expand_dims(Y, 0) - output, 2)
    # loss_eval = loss.eval()


    # Y = sess.run(y, feed_dict={X1: input})
    print((datetime.now() - t1).total_seconds())

    # writer = tf.summary.FileWriter(logdir='data', graph=sess.graph)
    # writer.close()
    #
    # tl = timeline.Timeline(run_metadata.step_stats)
    # trace = tl.generate_chrome_trace_format()
    # with open('timeline.json', 'w') as f:
    #     f.write(trace)


# output1 = np.expand_dims(Y, 0)

# z1.shape
# (10, 10, 5)

# z2.shape
# (1, 10, 10, 6)

# y.shape
# (10, 10, 64)