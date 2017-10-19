"""
Original source: https://github.com/simo23/conv2d_transpose/blob/master/test.py
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim as slim
import os


# Import the dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

tf.reset_default_graph()

with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, [None, 784], name="x-in")

with tf.name_scope('batchnorm_var'):
    batchnorm = tf.Variable(True, name='batchnorm')

# Batchnorm variable
epsilon = 1e-3


def weight_variable(shape):
    with tf.name_scope('weights'):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)


def bias_variable(shape):
    with tf.name_scope('biases'):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)


def conv2d(x, W):
    with tf.name_scope('conv2d'):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    with tf.name_scope('pooling'):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def max_pool_with_argmax(net, stride):
    with tf.name_scope('MaxPoolArgMax'):
        _, mask = tf.nn.max_pool_with_argmax(net, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1],
                                             padding='SAME')
        mask = tf.stop_gradient(mask)
        net = slim.max_pool2d(net, kernel_size=[stride, stride], stride=2)
        return net, mask


###############################################################################################################
##### NETWORK DEFINITION
###############################################################################################################

with tf.name_scope('reshape_vec2img'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

with tf.name_scope('conv1'):
    W_conv1 = weight_variable([7, 7, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = conv2d(x_image, W_conv1) + b_conv1

    if batchnorm == True:
        batch_mean1, batch_var1 = tf.nn.moments(h_conv1, [0])
        scale1 = tf.Variable(tf.ones([28, 28, 32]))
        beta1 = tf.Variable(tf.zeros([28, 28, 32]))
        batch_norm1 = tf.nn.batch_normalization(h_conv1, batch_mean1, batch_var1, beta1, scale1, epsilon)
        act_conv1 = tf.nn.relu(batch_norm1)
    else:
        act_conv1 = tf.nn.relu(h_conv1)

    h_pool1, mask = max_pool_with_argmax(act_conv1, 2)

with tf.name_scope('conv2'):
    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = conv2d(h_pool1, W_conv2) + b_conv2

    batch_mean2, batch_var2 = tf.nn.moments(h_conv2, [0])
    scale2 = tf.Variable(tf.ones([14, 14, 64]))
    beta2 = tf.Variable(tf.zeros([14, 14, 64]))

    batch_norm2 = tf.nn.batch_normalization(h_conv2, batch_mean2, batch_var2, beta2, scale2, epsilon)
    act_conv2 = tf.nn.relu(batch_norm2)
    h_pool2 = max_pool_2x2(act_conv2)

with tf.name_scope('reshape_pool2fc'):
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    z_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1

    batch_mean3, batch_var3 = tf.nn.moments(z_fc1, [0])
    scale3 = tf.Variable(tf.ones([1024]))
    beta3 = tf.Variable(tf.zeros([1024]))
    batch_norm3 = tf.nn.batch_normalization(z_fc1, batch_mean3, batch_var3, beta3, scale3, epsilon)
    h_fc1 = tf.nn.relu(z_fc1)

with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    out_y = tf.matmul(h_fc1, W_fc2) + b_fc2

####################################################################################################
### SESSION DEFINITION AND CHECKPOINT RESTORING
####################################################################################################

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

# Restore the checkpoint
saver = tf.train.Saver()
saver.restore(sess, './ckpt/model.ckpt')
print('Best model restored ...')


image = mnist.test.images[0]

# Compute the activations of the 32 filters in the first convolutional layer
n_filters = 32
activations = sess.run(h_conv1, feed_dict={x: np.reshape(image, [1, 784], order='C'), batchnorm: True})
transposed_weights = tf.transpose(W_conv1, perm=[0, 1, 3, 2])
flipped_weights = tf.reverse(transposed_weights,axis=[0,1])


# We expect the output of the deconvolution to be 1 image of 28x28x1
output_shape = [1, 28, 28, 1]
strides = [1, 1, 1, 1]

############################################################################################
# Make the deconvolution with just one filter's activation != 0 to see the filter's focus
#############################################################################################
chosen_filter = 5
for i in range(n_filters):
    if i != chosen_filter:
        activations[0,:,:,i] = 0




deconv_conv2d = sess.run(
    tf.cast(tf.nn.conv2d(activations, flipped_weights, strides=strides, padding='SAME'), tf.int64))
deconv_conv2d_backprop = sess.run(
    tf.cast(tf.nn.conv2d_backprop_input([1, 28, 28, 1], W_conv1, activations, strides=strides, padding='SAME'),
            tf.int64))
deconv_conv2d_transpose = sess.run(
    tf.cast(tf.nn.conv2d_transpose(activations, W_conv1, output_shape=output_shape, strides=strides, padding='SAME'),
            tf.int64))

fig = plt.figure(1, figsize=(30, 10))
n_columns = 5
n_rows = 1

image = sess.run(tf.reshape(image, [1, 28, 28, 1]))
plt.subplot(n_rows, n_columns, 1)
plt.title('Original Image')
plt.imshow(image[0, :, :, 0], interpolation="nearest", cmap="gray")

n_filter_to_show = chosen_filter
plt.subplot(n_rows, n_columns, 2)
plt.title('Original Activation (Filter #' + str(n_filter_to_show) + ')')
plt.imshow(activations[0, :, :, n_filter_to_show], interpolation="nearest", cmap="gray")

plt.subplot(n_rows, n_columns, 3)
plt.title('Conv2d Deconv')
plt.imshow(deconv_conv2d[0, :, :, 0], interpolation="nearest", cmap="gray")

plt.subplot(n_rows, n_columns, 4)
plt.title('Conv2d Backprop Deconv')
plt.imshow(deconv_conv2d_backprop[0, :, :, 0], interpolation="nearest", cmap="gray")

plt.subplot(n_rows, n_columns, 5)
plt.title('Conv2d Transpose Deconv')
plt.imshow(deconv_conv2d_transpose[0, :, :, 0], interpolation="nearest", cmap="gray")

fig.savefig('DeconvTest.png')
