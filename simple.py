"""
Simple Convolutional Neural Network Vizualization
Author: Csaba Nemes
This version: https://github.com/csnemes2/conv-net-viz
Original forked from: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
from helper import *
import os
import shutil

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
learning_rate = 0.001
num_steps = 100
batch_size = 128
display_step = 10

# Network Parameters
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

remembered_tensors_list = []


def remember_tensor(tensor,mask=None):
    global remembered_tensors_list
    operation = tensor.op
    print(tensor.name + ' remembered')
    # tensor, operation, reversed_tensor, mask, reversed_mask
    remembered_tensors_list.append([tensor, operation, None, mask, None])


def reverse_operation(tensor, operation, prev_tensor, mask):
    global batch_size
    print('Reversing ' + tensor.name)
    if operation.type == 'Relu':
        return tf.nn.relu(prev_tensor)
    elif operation.type == 'BiasAdd':
        return tf.nn.bias_add(prev_tensor, -operation.inputs[1])
    elif operation.type == 'Conv2D':
        orig_strides = operation.get_attr('strides')
        orig_padding = operation.get_attr('padding')
        orig_shape = operation.inputs[0].shape.as_list()
        orig_shape[0] = batch_size
        return tf.nn.conv2d_transpose(prev_tensor, operation.inputs[1], output_shape=orig_shape, strides=orig_strides,
                                      padding=orig_padding)
    elif operation.type == 'MaxPool':
        orig_strides = operation.get_attr('strides')
        return unpool(prev_tensor,mask,orig_strides)
        pass
    else:
        exit('ERROR: You did not specify a reverse operation for type= ' + operation.type)


def build_reverse_chain():
    global remembered_tensors_list
    print('\nBuilding reverse chain')
    last_tensor_shape = remembered_tensors_list[-1][0].shape
    last_tensor = tf.placeholder(tf.float32, last_tensor_shape)
    for i in reversed(remembered_tensors_list):
        i[2] = last_tensor
        last_tensor = reverse_operation(i[0], i[1], i[2], i[3])
    return last_tensor


def conv2d(x0, W, b, strides=1):
    global batch_size

    x1 = tf.nn.conv2d(x0, W, strides=[1, strides, strides, 1], padding='SAME')
    remember_tensor(x1)

    x2 = tf.nn.bias_add(x1, b)
    remember_tensor(x2)

    x3 = tf.nn.relu(x2)
    remember_tensor(x3)

    return x3


def maxpool2d(x, k=2):
    """
      Tensorflow default implementation does not provide gradient operation on max_pool_with_argmax
      Therefore, we use max_pool_with_argmax to extract mask and
      plain max_pool for, eeem... max_pooling.

      https: // github.com / yselivonchyk / Tensorflow_WhatWhereAutoencoder / blob / master / WhatWhereAutoencoder.py
    """
    _, mask = tf.nn.max_pool_with_argmax(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME')
    mask = tf.stop_gradient(mask)
    net =tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                      padding='SAME')
    remember_tensor(net,mask=mask)

    return net

def unpool(net, mask, strides):
    global batch_size
    """
      https: // github.com / yselivonchyk / Tensorflow_WhatWhereAutoencoder / blob / master / WhatWhereAutoencoder.py
    """
    with tf.name_scope('UnPool2D'):
        ksize=strides
        input_shape = net.get_shape().as_list()
        input_shape[0]=batch_size
        output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
        print('ksize='+str(ksize))
        print('input_shape='+str(input_shape))
        print('output_shape='+str(output_shape))
        # calculation indices for batch, height, width and feature maps
        one_like_mask = tf.ones_like(mask)
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
        feature_range = tf.range(output_shape[3], dtype=tf.int64)
        f = one_like_mask * feature_range
        # transpose indices & reshape update values to one dimension
        updates_size = tf.size(net)
        indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
        values = tf.reshape(net, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        return ret


# Create model
def conv_net(x, weights, biases, dropout):
    global remembered_tensors_list
    global input_viz
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    vx_shape = x.shape.as_list()
    vx_shape[0] = batch_size

    with tf.name_scope('layer1'):
        # Convolution Layer
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(conv1, k=2)

    input_viz = build_reverse_chain()

    with tf.name_scope('layer2'):
        # Convolution Layer
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


def max_activation_for_layers(tensor, top_num=128):
    print('Running through the test dataset')
    test_size = mnist.test.num_examples
    total_batch = int(test_size / batch_size)
    total_activation_data = np.zeros((total_batch * batch_size, tensor.shape[3]), dtype=np.float32)
    start = 0
    for i in range(total_batch):
        batch = mnist.test.next_batch(batch_size)
        tensor_data = sess.run(tensor, feed_dict={X: batch[0]})
        total_activation_data[start:start + batch_size, :] = np.sum(tensor_data, axis=(1, 2))
        start += batch_size
    print('total_activation_data.shape=' + str(total_activation_data.shape))
    # print(total_activation_data[:100,:10])
    top_indices = np.zeros((total_activation_data.shape[0], top_num), dtype=np.int32)
    for ch in range(total_activation_data.shape[1]):
        top_indices[ch, :] = sorted(range(len(total_activation_data[:, ch])),
                                    key=lambda i: total_activation_data[i, ch], reverse=True)[:top_num]
    return top_indices


def viz(sess, tensor_name):
    success = False
    for i in remembered_tensors_list:
        tensor = i[0]
        operation = i[1]
        reverse_tensor = i[2]
        if tensor.name == tensor_name:
            success = True

            top_indices = max_activation_for_layers(tensor)

            tensor_name_unix = tensor_name.replace('/', "").replace(':', "")
            print('Visualizing tensor' + tensor_name + '\t in folder=' + tensor_name_unix + '\tits shape=' + str(
                tensor.shape))
            target = 'simple_results/' + tensor_name_unix
            start_clean_dir(target)
            for channel_index in range(0, int(tensor.shape[3])):
                channel_top_indices = top_indices[channel_index, :]
                viz_channel(sess, target, tensor, reverse_tensor, channel_index, channel_top_indices)
    if not success:
        print('Error')
        print('Not found=' + tensor_name)
        print('In')
        for i in remembered_tensors_list:
            tensor = i[0]
            print('\t' + tensor.name)


def adjust_tensor_to_input(tensor_data, layer_index, image_shape):
    #print(image_shape)
    ts = tensor_data.shape
    #print(ts)
    adjusted_tensor_data = np.zeros(image_shape, dtype=np.float32)
    adjusted_tensor_data[:,:ts[1],:ts[2],0] = tensor_data[:,:,:,layer_index]
    return adjusted_tensor_data


def pimp_image_hist(data):
    a = np.amax(data)
    return data / a * 128 + 128


def viz_channel(sess, tensor_name, tensor, reverse_tensor, layer_index, channel_top_indices):
    global input_viz

    tensor_data = sess.run(tensor, feed_dict={X: mnist.test.images[channel_top_indices]})
    viz_top = 3
    print('\tVisualizing channel index' + str(layer_index))
    print('\t\tTop indices=' + str(channel_top_indices[:viz_top]))
    print('\t\tnp.max(tensor_data)='+str(np.max(tensor_data)))
    print('\t\tnp.min(tensor_data)=' + str(np.min(tensor_data)))

    new_tensor_data = np.zeros(tensor_data.shape, dtype=np.float32)
    new_tensor_data[:, :, :, layer_index] = tensor_data[:, :, :, layer_index]
    input_viz_to_save = sess.run(input_viz, feed_dict={reverse_tensor: new_tensor_data, X:mnist.test.images[channel_top_indices]})

    # display between 0 and 256
    input_viz_to_save = pimp_image_hist(input_viz_to_save)
    tensor_data = pimp_image_hist(tensor_data)

    adjusted_tensor_data = adjust_tensor_to_input(tensor_data, layer_index, input_viz_to_save.shape)

    save_viz_channel(tensor_name,
                     layer_index,
                     input_viz_to_save[:viz_top, :, :, :].reshape([viz_top * 28, 28]),
                     mnist.test.images[channel_top_indices[:viz_top]].reshape(viz_top * 28, 28) * 255,
                     adjusted_tensor_data[:viz_top,:,:,:].reshape([viz_top * 28, 28]))


def start_clean_dir(target):
    if os.path.exists(target):
        shutil.rmtree(target)
    if not os.path.exists(target):
        os.makedirs(target)


def save_viz_channel(target_dir, lidx, viz_pics, orig_pics, layer_pics):
    pic = np.concatenate((orig_pics, layer_pics, viz_pics), axis=0)
    save_pic(target_dir + '/' + str(lidx)+'.bmp', pic)


with tf.Session() as sess:
    global input_viz
    sess.run(init)

    for step in range(1, num_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    print(mnist.train.images.shape)
    print(mnist.test.images.shape)
    print(mnist.validation.images.shape)

    print(np.min(mnist.train.images[0]))
    print(np.max(mnist.train.images[0]))

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                        Y: mnist.test.labels[:256],
                                        keep_prob: 1.0}))
    viz(sess, 'layer1/Conv2D:0')
    viz(sess, 'layer1/BiasAdd:0')
    viz(sess, 'layer1/Relu:0')
    viz(sess, 'layer1/MaxPool:0')
