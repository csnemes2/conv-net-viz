"""
Simple Convolutional Neural Network Vizualization
Author: Csaba Nemes
This version: https://github.com/csnemes2/conv-net-viz
Original forked from: https://github.com/aymericdamien/TensorFlow-Examples/
http://cv-tricks.com/image-segmentation/transpose-convolution-in-tensorflow/
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
num_steps = 10000
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
remembered_reception_sizes = dict()


def compute_receptive_field(tensor):
    # assumtion parent tensor should have been remembered already
    # lazy upper bound
    # kernel: +(kernel-1)
    # stride: xsstide
    prec = 1
    conw = 1
    for i in tensor.op.inputs:
        # for j in remembered_tensors_list:
        #    if i == j[0]:
        #        prec = j[4]
        if i.name in remembered_reception_sizes:
            prec = remembered_reception_sizes[i.name]
            print("parent found with receptive field=" + str(prec))
        # hack for now
        if i.op.type == 'Identity':
            conw = int(i.shape[0])

    if tensor.op.type == 'Conv2D':
        prec = prec + (conw - 1)
    try:
        orig_strides = tensor.op.get_attr('strides')
        prec = prec * orig_strides[1]
    except:
        pass

    return prec


def remember_tensor(tensor, mask=None):
    global remembered_tensors_list, remembered_reception_sizes
    operation = tensor.op
    receptive_field = compute_receptive_field(tensor)
    print(tensor.name + ' remembered' + ' receptive_field=' + str(receptive_field))
    remembered_reception_sizes[tensor.name] = receptive_field
    # 0:tensor, 1:operation, 2:reversed_tensor, 3:mask,
    remembered_tensors_list.append([tensor, operation, None, mask])


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
        return unpool(prev_tensor, mask, orig_strides)
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
    with tf.name_scope('Pool2D'):
        _, mask = tf.nn.max_pool_with_argmax(
            x,
            ksize=[1, k, k, 1],
            strides=[1, k, k, 1],
            padding='SAME')
        mask = tf.stop_gradient(mask)
        net = tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                             padding='SAME')
        remember_tensor(net, mask=mask)

    return net


def unpool(net, mask, strides):
    global batch_size
    """
      https: // github.com / yselivonchyk / Tensorflow_WhatWhereAutoencoder / blob / master / WhatWhereAutoencoder.py
    """
    with tf.name_scope('UnPool2D'):
        ksize = strides
        input_shape = net.get_shape().as_list()
        input_shape[0] = batch_size
        output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
        print('ksize=' + str(ksize))
        print('input_shape=' + str(input_shape))
        print('output_shape=' + str(output_shape))
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

    with tf.name_scope('layer2'):
        # Convolution Layer
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2)

    input_viz = build_reverse_chain()

    with tf.name_scope('layer3'):
        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)

    with tf.name_scope('layer4'):
        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out


with tf.name_scope('variables'):
    # Store layers weight & bias
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([7, 7, 1, 32])),
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

with tf.name_scope('training'):
    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

with tf.name_scope('score'):
    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


def max_activation_for_layers(layer, top_num=128, mode='sum'):
    print('Running through the test dataset')
    test_size = mnist.test.num_examples
    total_batch = int(test_size / batch_size)
    num_of_images = total_batch * batch_size
    num_of_channels = layer.shape[3]
    sum_activation_data = np.zeros((num_of_images, num_of_channels), dtype=np.float32)
    max_activation_data = np.zeros((num_of_images, num_of_channels), dtype=np.float32)

    for i in xrange(total_batch):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        batch = mnist.test.images[batch_start:batch_end]
        tensor_data = sess.run(layer, feed_dict={X: batch})
        sum_activation_data[batch_start:batch_end, :] = np.sum(tensor_data, axis=(1, 2))
        max_activation_data[batch_start:batch_end, :] = np.max(tensor_data, axis=(1, 2))

    print(str(layer) + ' sum_activation_data.shape=' + str(sum_activation_data.shape))

    top_indices = np.zeros((num_of_images, top_num), dtype=np.int32)
    for ch in xrange(num_of_channels):
        activation_base = sum_activation_data
        if mode == 'max':
            activation_base = max_activation_data
        top_indices[ch, :] = sorted(range(num_of_images),
                                    key=lambda i: activation_base[i, ch], reverse=True)[:top_num]

    if False:
        # for ch in xrange(num_of_channels):
        for ch in xrange(1):
            batch_indices = top_indices[ch, :]
            batch = mnist.test.images[batch_indices]
            tensor_data = sess.run(layer, feed_dict={X: batch})
            for j in xrange(batch_size):
                t = tensor_data[j, :, :, ch]
                print(j, batch_indices[j], np.sum(t), np.max(t), np.min(t))

    return top_indices


def tensor_pretty_name(tensor_name):
    return tensor_name.replace('/', "").replace(':', "")


def viz(sess, tensor_name, mode='sum'):
    success = False
    for i in remembered_tensors_list:
        tensor = i[0]
        operation = i[1]
        reverse_tensor = i[2]
        num_of_channels = tensor.shape[3]
        if tensor.name == tensor_name:
            success = True

            top_indices = max_activation_for_layers(tensor, mode=mode)

            tensor_name_unix = tensor_pretty_name(tensor_name)+mode
            print('Visualizing tensor' + tensor_name + '\t in folder=' + tensor_name_unix + '\tits shape=' + str(
                tensor.shape) + " Activation calc mode="+mode)
            target = 'simple_results/' + tensor_name_unix
            start_clean_dir(target)

            for channel_index in xrange(num_of_channels):
                if mode == 'fix':
                    channel_top_indices = list(xrange(128))
                else:
                    channel_top_indices = top_indices[channel_index, :]
                viz_channel(sess, target, tensor, reverse_tensor, channel_index, channel_top_indices)
            viz_channel(sess, target, tensor, reverse_tensor, None, list(xrange(128)))
    if not success:
        print('Error')
        print('Not found=' + tensor_name)
        print('In')
        for i in remembered_tensors_list:
            tensor = i[0]
            print('\t' + tensor.name)


def adjust_tensor_to_input(tensor_data, ch_index, image_shape):
    # print(image_shape)
    ts = tensor_data.shape
    # print(ts)
    adjusted_tensor_data = np.ones(image_shape, dtype=np.float32)*255
    if ch_index is None:
        # adjusted_tensor_data[:, :ts[1], :ts[2], 0] = np.sum(tensor_data[:, :, :, :], axis=3)
        adjusted_tensor_data[:, :ts[1], :ts[2], 0] = tensor_data[:, :, :, 0]
    else:
        adjusted_tensor_data[:, :ts[1], :ts[2], 0] = tensor_data[:, :, :, ch_index]
    return adjusted_tensor_data


def image_norm1(data):
    amin = np.min(data)
    positive = data + np.abs(amin)
    amax = np.max(positive)
    return (positive) / amax * 255


def image_norm2(data):
    amax = np.max(data)
    return (data) / amax * 255


def image_norm3(data):
    for i in xrange(128):
        amax = np.max(data[i, :, :, :])
        data[i, :, :, :] = data[i, :, :, :] / amax * 255
    return data


def image_norm4(data):
    for i in xrange(128):
        amin = np.min(data[i, :, :, :])
        positive = data[i, :, :, :] + np.abs(amin)
        amax = np.max(positive)
        data[i, :, :, :] = positive / amax * 255

    return data


def viz_channel(sess, tensor_name, tensor, reverse_tensor, ch_index, channel_top_indices):
    """
    Vizualization for one channel  in the tensor
    :param sess:
    :param tensor_name:
    :param tensor:
    :param reverse_tensor:
    :param ch_index: channel index, if None, everything is reconstructed
    :param channel_top_indices: 1st channel, 2nd indices
    :return:
    """
    global input_viz, remembered_reception_sizes

    tensor_data = sess.run(tensor, feed_dict={X: mnist.test.images[channel_top_indices]})
    viz_top = 10
    print('\t Layer=' + str(tensor))
    print('\tVisualizing channel index' + str(ch_index))
    print('\t\tTop indices=' + str(channel_top_indices[:viz_top]))
    print('\t\tnp.max(tensor_data)=' + str(np.max(tensor_data[0, :, :, :])))
    print('\t\tnp.min(tensor_data)=' + str(np.min(tensor_data[0, :, :, :])))

    if False:
        for j in xrange(128):
            t = tensor_data[j, :, :, ch_index]
            print(j, channel_top_indices[j], np.sum(t), np.max(t), np.min(t))

    new_tensor_data = np.zeros(tensor_data.shape, dtype=np.float32)
    if ch_index is not None:
        new_tensor_data[:, :, :, ch_index] = tensor_data[:, :, :, ch_index]
    else:
        new_tensor_data = tensor_data
    input_viz_to_save = sess.run(input_viz,
                                 feed_dict={reverse_tensor: new_tensor_data, X: mnist.test.images[channel_top_indices]})

    print('\t\tnp.max(input_viz_to_save)=' + str(np.max(input_viz_to_save[0, :, :, 0])))
    print('\t\tnp.min(input_viz_to_save)=' + str(np.min(input_viz_to_save[0, :, :, 0])))

    normed_tensor_data = image_norm4(tensor_data)
    adjusted_tensor_data = adjust_tensor_to_input(normed_tensor_data, ch_index, input_viz_to_save.shape)

    input_viz_to_save = image_norm4(input_viz_to_save)

    save_viz_channel(tensor_name,
                     remembered_reception_sizes[tensor.name],
                     ch_index,
                     input_viz_to_save[:viz_top, :, :, :],
                     mnist.test.images[channel_top_indices[:viz_top]].reshape([viz_top, 28, 28, 1]),
                     adjusted_tensor_data[:viz_top, :, :, :])


def start_clean_dir(target):
    if os.path.exists(target):
        shutil.rmtree(target)
    if not os.path.exists(target):
        os.makedirs(target)


def max_in_2D(pics2D):
    return np.unravel_index(np.argmax(pics2D), pics2D.shape)


def max_in_4D(pics4D):
    ret = []
    for i in xrange(pics4D.shape[0]):
        ret.append(max_in_2D(pics4D[i, :, :, 0]))
    return ret


def center_on(img, xys, reception_size):
    # lazy overbound
    wing_rec = int(int(reception_size) / 2)
    total_rec = int(wing_rec * 2) + 1

    # dontcare mode
    if total_rec > img.shape[1] / 2:
        return img

    ret_shape = list(img.shape)
    ret_shape[1] = total_rec
    ret_shape[2] = total_rec
    ret = np.zeros(ret_shape)

    for i in xrange(ret_shape[0]):
        (x, y) = xys[i]

        #print(x, y)

        p_xstart = x - wing_rec
        p_ystart = y - wing_rec
        p_xend = x + wing_rec
        p_yend = y + wing_rec

        # if p_xstart < 0:
        #     xstart = 0
        #     xpad = -p_xstart
        # else:
        #     xstart = p_xstart
        #     xpad = 0

        xstart = np.max([0, p_xstart])
        xpad = np.max([0, -p_xstart])

        # if p_ystart < 0:
        #     ystart = 0
        #     ypad = -p_ystart
        # else:
        #     ystart = p_ystart
        #     ypad = 0

        ystart = np.max([0, p_ystart])
        ypad = np.max([0, -p_ystart])

        xlen = np.min([img.shape[1], p_xend]) - xstart
        ylen = np.min([img.shape[2], p_yend]) - ystart

        # import ipdb;ipdb.set_trace()

        ret[i, xpad:xpad + xlen, ypad:ypad + ylen, 0] = img[i, xstart:xstart + xlen, ystart:ystart + ylen, 0]

    return ret


def save_viz_channel_max(target_dir, reception_size, lidx, viz_pics, orig_pics, layer_pics):
    print('\t\treception_size=' + str(reception_size))
    xys = max_in_4D(layer_pics)

    c_viz_pics = center_on(viz_pics, xys, reception_size)
    c_orig_pics = center_on(orig_pics, xys, reception_size)
    c_layer_pics = center_on(layer_pics, xys, reception_size)

    save_viz_channel_sum(target_dir, lidx, c_viz_pics, c_orig_pics, c_layer_pics, file_append='patch')


def save_viz_channel_sum(target_dir, lidx, viz_pics, orig_pics, layer_pics, file_append=''):
    xdim=viz_pics.shape[1]
    ydim=viz_pics.shape[2]
    c_viz_pics = viz_pics.reshape([viz_pics.shape[0] * xdim, ydim])
    c_orig_pics = orig_pics.reshape([orig_pics.shape[0] * xdim, ydim]) * 255
    c_layer_pics = layer_pics.reshape([layer_pics.shape[0] * xdim, ydim])
    pic = np.concatenate((c_orig_pics, c_layer_pics, c_viz_pics), axis=0)
    save_pic(target_dir + '/' + str(lidx)+file_append, pic)


def save_viz_channel(target_dir, reception_size, lidx, viz_pics, orig_pics, layer_pics):
    save_viz_channel_sum(target_dir, lidx, viz_pics, orig_pics, layer_pics)
    save_viz_channel_max(target_dir, reception_size, lidx, viz_pics, orig_pics, layer_pics)


with tf.Session() as sess:
    global input_viz
    sess.run(init)

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("simple_log", sess.graph)
    saver = tf.train.Saver()

    ckpt_file = './simple.ckpt'
    if os.path.isfile(ckpt_file + '.index'):
        saver.restore(sess, ckpt_file)
    else:
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
        saver.save(sess, ckpt_file)

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                        Y: mnist.test.labels[:256],
                                        keep_prob: 1.0}))
    #layer1
    if True:
        viz(sess, 'layer1/Conv2D:0', mode='max')
        viz(sess, 'layer1/BiasAdd:0', mode='max')
        viz(sess, 'layer1/Relu:0', mode='max')
        viz(sess, 'layer1/Pool2D/MaxPool:0', mode='max')

        viz(sess, 'layer1/Conv2D:0', mode='sum')
        viz(sess, 'layer1/BiasAdd:0', mode='sum')
        viz(sess, 'layer1/Relu:0', mode='sum')
        viz(sess, 'layer1/Pool2D/MaxPool:0', mode='sum')

        viz(sess, 'layer1/Conv2D:0', mode='fix')
        viz(sess, 'layer1/BiasAdd:0', mode='fix')
        viz(sess, 'layer1/Relu:0', mode='fix')
        viz(sess, 'layer1/Pool2D/MaxPool:0', mode='fix')

    # layer2
    if True:
        viz(sess, 'layer2/Conv2D:0', mode='max')
        viz(sess, 'layer2/BiasAdd:0', mode='max')
        viz(sess, 'layer2/Relu:0', mode='max')
        viz(sess, 'layer2/Pool2D/MaxPool:0', mode='max')

        viz(sess, 'layer2/Conv2D:0', mode='sum')
        viz(sess, 'layer2/BiasAdd:0', mode='sum')
        viz(sess, 'layer2/Relu:0', mode='sum')
        viz(sess, 'layer2/Pool2D/MaxPool:0', mode='sum')

        viz(sess, 'layer2/Conv2D:0', mode='fix')
        viz(sess, 'layer2/BiasAdd:0', mode='fix')
        viz(sess, 'layer2/Relu:0', mode='fix')
        viz(sess, 'layer2/Pool2D/MaxPool:0', mode='fix')
