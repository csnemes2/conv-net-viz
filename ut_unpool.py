import numpy as np
import tensorflow as tf


def print_mat(mat, nam=''):
    print(nam)
    for i in mat:
        print (i[:, :, 0])
        print (i[:, :, 1])


batch_size = 3
def maxpool2d(x, ksize, strides, padding):
    #
    # mask with correct argmax indices
    # currently batch size is needed
    #
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

    return net, mask

def unpool(net, mask, ksize, strides, padding, orig_input_shape):
    #
    # ksize: window size
    # strides: strides, aka stepping
    # padding:
    # orig_out_shape: if padding is VALID, easier to know the input shape
    #
    with tf.name_scope('UnPool2D'):
        print('\t\tUnpooling for tensor=' + net.name)
        orig_out_shape = net.get_shape().as_list()
        orig_out_shape[0] = batch_size
        orig_input_shape[0] = batch_size
        updates_size = tf.size(net)

        print('\t\t padding=' + padding)
        print('\t\t mask=' + str(mask.get_shape().as_list()))
        print('\t\t ksize=' + str(ksize))
        print('\t\t strides=' + str(strides))
        print('\t\t orig_input_shape=' + str(orig_input_shape))
        print('\t\t orig_out_shape=' + str(orig_out_shape))

        if padding == 'SAME':
            new_output_shape = (
                orig_out_shape[0], orig_out_shape[1] * strides[1],
                orig_out_shape[2] * strides[2],
                orig_out_shape[3])
            print('\t\t new_output_shape=' + str(new_output_shape))

        elif padding == 'VALID':
            # output_spatial_shape[i] = ceil((input_spatial_shape[i]
            # - (spatial_filter_shape[i]-1) * dilation_rate[i]) / strides[i]).
            new_output_shape = orig_input_shape
            print('\t\t new_output_shape=' + str(new_output_shape))
            assert np.ceil(float(orig_input_shape[1] - ksize[1] + 1) / strides[1]) == \
                   orig_out_shape[1]

        #
        #   problem: tf.scatter_nd use add if multiple indices refer to the same
        #   solution: I count multiple indices, and divide the values vector
        #
        mask_list = tf.reshape(mask, [updates_size])
        val_list = tf.reshape(net, [updates_size])
        u_mask, u_idx, u_count = tf.unique_with_counts(mask_list)
        div_list = tf.gather(u_count, u_idx)
        div = tf.reshape(div_list, orig_out_shape)

        val_list = tf.cast(val_list, tf.float32) / tf.cast(div_list, tf.float32)

        # Using:
        #  [b, y, x, c]
        # flattened index ((b * height + y) * width + x) * channels + c.
        img = mask // (new_output_shape[3] * new_output_shape[2] * new_output_shape[1])
        y = mask % (new_output_shape[3] * new_output_shape[2] * new_output_shape[1]) // (new_output_shape[3] * new_output_shape[2])
        x = mask % (new_output_shape[3] * new_output_shape[2]) // new_output_shape[3]
        ch = mask % new_output_shape[3]
        indices = tf.transpose(tf.reshape(tf.stack([img, y, x, ch]), [4, updates_size]))

        ret = tf.scatter_nd(indices, val_list, new_output_shape)
        return ret, div


sh = [3, 5, 5, 2]
mat = np.zeros(sh)
ch1 = mat[0, :, :, 0]
ch2 = mat[0, :, :, 1]
ch3 = mat[1, :, :, 0]
ch4 = mat[1, :, :, 1]
ch5 = mat[2, :, :, 0]
ch6 = mat[2, :, :, 1]

ch1[0, 0] = 7
ch1[2, 1] = 9
ch2[1, 1] = 5
ch3[1, 1] = 5
ch5[3, 3] = 1

x = tf.placeholder(tf.float32, sh)
x2, mask = maxpool2d(x, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')
ph = tf.placeholder(tf.float32, x2.get_shape().as_list())
x3, mm = unpool(ph, mask, [1, 3, 3, 1], [1, 2, 2, 1],
                'VALID', x.get_shape().as_list())

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

print_mat(mat, 'orig')
mat2, mask2 = sess.run([x2, mask], feed_dict={x: mat})

print_mat(mat2, 'pooled')
print_mat(mask2, 'mask')

# un pool
mat3, divider = sess.run([x3, mm], feed_dict={x: mat, ph: mat2})

print_mat(mat3,'unpooled')
print_mat(divider,'divider')
