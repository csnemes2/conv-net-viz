################################################################################
#Michael Guerzhoy and Davi Frossard, 2016
#AlexNet implementation in TensorFlow, with weights
#Details: 
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

from numpy import *
import os
#from pylab import *
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
from lib import viz
from lib import imagenet
import uuid

import tensorflow as tf

from caffe_classes import class_names

train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]

b_local_respose = True
################################################################################
#Read Image, and change to BGR


im1 = (imread("laska.png")[:,:,:3]).astype(float32)
im1 = im1 - mean(im1)
im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]

im2 = (imread("poodle.png")[:,:,:3]).astype(float32)
im2[:, :, 0], im2[:, :, 2] = im2[:, :, 2], im2[:, :, 0]

DB = imagenet.ImageDB("/home/csn/IMAGENET/2012_img_val_227")
DB.limit_len(8192)

x = tf.placeholder(tf.float32, (None,) + xdim)
DV = viz.DeconvVisualization(batch_size=128,target_dir="alex_results", input_ph=x, test_images=DB, max_channel_num= 6)

################################################################################

# (self.feed('data')
#         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
#         .lrn(2, 2e-05, 0.75, name='norm1')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
#         .lrn(2, 2e-05, 0.75, name='norm2')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#         .conv(3, 3, 384, 1, 1, name='conv3')
#         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
#         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
#         .fc(4096, name='fc6')
#         .fc(4096, name='fc7')
#         .fc(1000, relu=False, name='fc8')
#         .softmax(name='prob'))

#In Python 3.5, change this to:
net_data = load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
#net_data = load("bvlc_alexnet.npy").item()

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1, do_viz=False):
    global DV
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)


    if group==1:
        conv = convolve(input, kernel)
        if do_viz:
            DV.remember_tensor(conv)
    else:
        input_groups =  tf.split(input, group, 3)   #tf.split(3, group, input)
        kernel_groups = tf.split(kernel, group, 3)  #tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        if do_viz:
            for g in input_groups:
                DV.remember_tensor(g)
            same_level_hash = uuid.uuid4()
            for g in output_groups:
                DV.remember_tensor(g,hash_list=[same_level_hash])
        conv = tf.concat(output_groups, 3)          #tf.concat(3, output_groups)
        if do_viz:
            DV.remember_tensor(conv)
    tmp = tf.nn.bias_add(conv, biases)
    if do_viz:
        DV.remember_tensor(tmp)
    #tmp = tf.reshape(tmp, [-1]+conv.get_shape().as_list()[1:])
    #DV.remember_tensor(tmp)
    return tmp

def maxpool2d(x, ksize, strides,padding):
    """
    Originally from:      https: // github.com / yselivonchyk / Tensorflow_WhatWhereAutoencoder / blob / master / WhatWhereAutoencoder.py
    Then: scatter_nd related is bug corrected
    """
    batch_size = DV.batch_size
    orig_input_shape = x.get_shape().as_list()
    orig_input_shape[0] = batch_size


    with tf.name_scope('Pool2D'):
        _, mask = tf.nn.max_pool_with_argmax(
            x,
            ksize=ksize,
            strides=strides,
            padding=padding)
        mask = tf.stop_gradient(mask)
        net = tf.nn.max_pool(x, ksize=ksize, strides=strides,
                             padding=padding)

        # note the documentation is not correct
        #   https: // github.com / tensorflow / tensorflow / pull / 7161
        #
        # i will force what the documentation says:
        #   [b, y, x, c]
        # flattened index ((b * height + y) * width + x) * channels + c.
        delta = orig_input_shape[1] * orig_input_shape[2] * orig_input_shape[3]
        correct_mask = tf.reshape(
            tf.range(start=0, limit=delta * batch_size, delta=delta,
                     dtype=tf.int64),
            shape=[batch_size, 1, 1, 1])
        mask = mask + correct_mask

        DV.remember_tensor(net, mask=mask)

    return net



with tf.name_scope('layer1'):
    #conv1
    #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1W = tf.Variable(net_data["conv1"][0])
    conv1b = tf.Variable(net_data["conv1"][1])
    conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1, do_viz= True)
    conv1 = tf.nn.relu(conv1_in)
    DV.remember_tensor(conv1)

    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    if b_local_respose:
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                          depth_radius=radius,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          bias=bias)
        DV.remember_tensor(lrn1)
    else:
        lrn1=conv1

    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    #maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    maxpool1 = maxpool2d(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


with tf.name_scope('layer2'):
    #conv2
    #conv(5, 5, 256, 1, 1, group=2, name='conv2')
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2W = tf.Variable(net_data["conv2"][0])
    conv2b = tf.Variable(net_data["conv2"][1])
    conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group,  do_viz= True)
    conv2 = tf.nn.relu(conv2_in)
    DV.remember_tensor(conv2)

    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    if b_local_respose:
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                          depth_radius=radius,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          bias=bias)
        DV.remember_tensor(lrn2)
    else:
        lrn2=conv2

    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    #maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    maxpool2 = maxpool2d(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

with tf.name_scope('layer3'):
    #conv3
    #conv(3, 3, 384, 1, 1, name='conv3')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3W = tf.Variable(net_data["conv3"][0])
    conv3b = tf.Variable(net_data["conv3"][1])
    conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group, do_viz=True)
    conv3 = tf.nn.relu(conv3_in)
    DV.remember_tensor(conv3)



with tf.name_scope('layer4'):
    #conv4
    #conv(3, 3, 384, 1, 1, group=2, name='conv4')
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4W = tf.Variable(net_data["conv4"][0])
    conv4b = tf.Variable(net_data["conv4"][1])
    conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group, do_viz=True)
    conv4 = tf.nn.relu(conv4_in)
    DV.remember_tensor(conv4)


with tf.name_scope('layer5'):
    #conv5
    #conv(3, 3, 256, 1, 1, group=2, name='conv5')
    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv5W = tf.Variable(net_data["conv5"][0])
    conv5b = tf.Variable(net_data["conv5"][1])
    conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group, do_viz=True)
    conv5 = tf.nn.relu(conv5_in)
    DV.remember_tensor(conv5)

    #maxpool5
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    #maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)
    maxpool5 = maxpool2d(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

input_viz = DV.build_reverse_chain()

with tf.name_scope('layer6'):
    #fc6
    #fc(4096, name='fc6')
    fc6W = tf.Variable(net_data["fc6"][0])
    fc6b = tf.Variable(net_data["fc6"][1])
    fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

with tf.name_scope('layer7'):
    #fc7
    #fc(4096, name='fc7')
    fc7W = tf.Variable(net_data["fc7"][0])
    fc7b = tf.Variable(net_data["fc7"][1])
    fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

with tf.name_scope('layer8'):
    #fc8
    #fc(1000, relu=False, name='fc8')
    fc8W = tf.Variable(net_data["fc8"][0])
    fc8b = tf.Variable(net_data["fc8"][1])
    fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)


#prob
#softmax(name='prob'))
prob = tf.nn.softmax(fc8)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter("alex_log", sess.graph)

#DV.print_available_tensors()

#DV.viz(sess, 'layer1/Conv2D:0', mode='max')
#DV.viz(sess, 'layer1/Pool2D/MaxPool:0', mode='max')

#DV.viz(sess, 'layer2/Conv2D:0', mode='max')
#DV.viz(sess, 'layer2/Conv2D_1:0', mode='max')

# here you have group=1 again, so just one conv
#DV.viz(sess, 'layer3/Conv2D:0', mode='max')

#DV.viz(sess, 'layer4/Conv2D:0', mode='max')
#DV.viz(sess, 'layer4/Conv2D_1:0', mode='max')

DV.viz(sess, 'layer5/Conv2D:0', mode='max')
DV.viz(sess, 'layer5/Conv2D_1:0', mode='max')

exit()
t = time.time()
output = sess.run(prob, feed_dict = {x:[im1,im2]})
################################################################################

#Output:


for input_im_ind in range(output.shape[0]):
    inds = argsort(output)[input_im_ind,:]
    print("Image", input_im_ind)
    for i in range(5):
        print(class_names[inds[-1-i]], output[input_im_ind, inds[-1-i]])

print(time.time()-t)
