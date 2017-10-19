"""
 test for conv2d and conv2d_transpose
 sources:
  https://github.com/simo23/conv2d_transpose/blob/master/test.py
  https://gist.github.com/yxlao/ef50416011b9587835ac752aa3ce3530

 deconvolution is basically the "backward pass"
 if you accept that a convolution after (image reshaping) is just a big matrix multiplication
 then deconvolution is just the transpose of that matrix
 which is again a convolution
 however the road to the big matrix is tricky in most cases, paddings etc.
 so to find out which convolution fits to the transposed big matrix is hard
 in very simple case it is easy:
    - just flip the kernel if padding is SAME
    - flip the kernel and padd the input if padding is VALID
    - do more if the kernel size is even

 plus, if you have more channels then you have to tranpose the kernel a bit

 finally, why is deconvolution used for vizualization:

 input->conv->feature

 loss=fun(feature)
 delta loss/delta input = delta loss/delta feature * delta feature/delta input

"""

import tensorflow as tf
import numpy as np


def plot_tensor(tensor, title):
    print (title)
    # [batch, in_height, in_width, in_channels]
    pass
    g = ((b, c)
         for b in xrange(tensor.shape[0])
         for c in xrange(tensor.shape[3]))
    for (b, c) in g:
        print(str(tensor[b, :, :, c]) + ' pic ' + str(b) + ' channel ' + str(c))


def plot_filter(filt, title):
    print (title)
    # [filter_height, filter_width, in_channels, out_channels]
    g = ((i, o)
         for o in xrange(filt.shape[3])
         for i in xrange(filt.shape[2]))
    for (i, o) in g:
        print(str(filt[:, :, i, o]) + ' from ch ' + str(i) + ' into ch ' + str(o))


def tf_pad_to_full_conv2d(x, w_size):
    return tf.pad(x, [[0, 0],
                      [w_size - 1, w_size - 1],
                      [w_size - 1, w_size - 1],
                      [0, 0]])


def run2(pad = 'SAME'):
    x_size = 5
    w_size = 3  # in not even case the padding is not symmetric, don't try
    x_shape = (1, x_size, x_size, 1)
    w_shape = (w_size, w_size, 1, 1)
    strides = (1, 1, 1, 1)

    x_np = np.random.randint(10, size=x_shape)
    w_np = np.random.randint(10, size=w_shape)

    # tf forward
    x = tf.constant(x_np, dtype=tf.float32)
    w = tf.constant(w_np, dtype=tf.float32)
    out = tf.nn.conv2d(input=x, filter=w, strides=strides, padding=pad)

    d_x_transpose = tf.nn.conv2d_transpose(value=out,
                                           filter=w,
                                           output_shape=x_shape,
                                           strides=strides,
                                           padding=pad)
    w_flip = tf.reverse(w, axis=[0, 1])

    if pad =='VALID':
        out= tf_pad_to_full_conv2d(out, w_size)

    d_x_manual = tf.nn.conv2d(input=out,
                              filter=w_flip,
                              strides=strides,
                              padding=pad)



    session = tf.Session()
    tf.global_variables_initializer()

    plot_tensor(session.run(x), "INPUT")
    plot_filter(session.run(w), "FILTER")
    plot_tensor(session.run(out), "CONVOLUTION")
    plot_filter(session.run(w_flip), "FILTER TRANSPOSE")
    plot_tensor(session.run(d_x_manual), "TRANSPOSE MANUAL")
    plot_tensor(session.run(d_x_transpose), "TRANSPOSE BUILTIN")


run2('SAME')
run2('VALID')
