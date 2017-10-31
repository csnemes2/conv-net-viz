from __future__ import division, print_function, absolute_import

import tensorflow as tf
from helper import *
import os
import shutil
import scipy.ndimage as ndimage


def center_on(img, xys, reception_size):
    wing_rec = int(int(reception_size) / 2)
    total_rec = int(wing_rec * 2) + 1

    # dontcare mode
    if total_rec > img.shape[1]:
        return img

    ret_shape = list(img.shape)
    ret_shape[1] = total_rec
    ret_shape[2] = total_rec
    ret = np.zeros(ret_shape)

    for i in xrange(ret_shape[0]):
        (x, y) = xys[i]

        # print(x, y)

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

        ret[i, xpad:xpad + xlen, ypad:ypad + ylen, :] = img[i, xstart:xstart + xlen,
                                                        ystart:ystart + ylen, :]

    return ret


def color_on(img, xys, reception_size, pixel_value=1, color_dim=0):
    img = convert_4D_RGB(img)

    # print ('color_on max=',np.max(img))

    wing_rec = int(int(reception_size) / 2)
    total_rec = int(wing_rec * 2) + 1

    # dontcare mode
    if total_rec > img.shape[1]:
        return img

    ret_shape = list(img.shape)
    ret_shape[1] = total_rec
    ret_shape[2] = total_rec
    ret = np.zeros(ret_shape)

    for i in xrange(ret_shape[0]):
        (x, y) = xys[i]

        # print(x, y)

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


        # img[i, xstart:xstart + xlen, ystart:ystart + ylen, 0]=255
        for color in [0, 1, 2]:
            if color == color_dim:
                val = pixel_value
            else:
                val = 0
            img[i, xstart:xstart + xlen, ystart, color] = val
            img[i, xstart:xstart + xlen, ystart + ylen - 1, color] = val
            img[i, xstart, ystart:ystart + ylen, color] = val
            img[i, xstart + xlen - 1, ystart:ystart + ylen, color] = val

    return img


class DeconvVisualization:
    def __init__(self, batch_size=9, target_dir="results", input_ph=None,
                 test_images=None, viz_matrix_size=9, max_channel_num=10):
        self.batch_size = batch_size
        self.target_dir = target_dir
        self.input_ph = input_ph
        self.test_images = test_images
        if self.test_images:
            print("len test images=", self.test_images.len())
        self.remembered_tensors_list = []
        self.remembered_reception_sizes = dict()
        self.input_viz = None
        self.viz_matrix_size = viz_matrix_size
        assert self.batch_size >= self.viz_matrix_size  # for demo purpose that is enough
        self.max_channel_num = max_channel_num

    def add(self, x):
        self.data.append(x)

    def compute_receptive_field(self, tensor):
        # assumption parent tensor should have been remembered already
        # lazy upper bound
        # kernel: +(kernel-1)
        # stride: x stride
        prec = 1
        conw = 1
        for i in tensor.op.inputs:
            # for j in remembered_tensors_list:
            #    if i == j[0]:
            #        prec = j[4]
            if i.name in self.remembered_reception_sizes:
                prec = self.remembered_reception_sizes[i.name]
                print("parent found with receptive field=" + str(prec))
            else:
                print(
                    "parent not found with receptive field, its parent is the input image")

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

    def remember_tensor(self, tensor, mask=None):
        operation = tensor.op
        receptive_field = self.compute_receptive_field(tensor)
        print(tensor.name + ' remembered' + ' receptive_field=' + str(receptive_field))
        self.remembered_reception_sizes[tensor.name] = receptive_field
        # 0:tensor, 1:operation, 2:reversed_tensor, 3:mask,
        self.remembered_tensors_list.append([tensor, operation, None, mask])

    def reverse_operation(self, tensor, operation, prev_tensor, mask):
        print('\tReversing ' + tensor.name)
        if operation.type == 'Relu':
            return tf.nn.relu(prev_tensor)
        elif operation.type == 'BiasAdd':
            # return tf.nn.bias_add(prev_tensor, -operation.inputs[1])
            return prev_tensor
        elif operation.type == 'Conv2D':
            orig_strides = operation.get_attr('strides')
            orig_padding = operation.get_attr('padding')
            orig_shape = operation.inputs[0].shape.as_list()
            orig_shape[0] = self.batch_size
            return tf.nn.conv2d_transpose(prev_tensor, operation.inputs[1],
                                          output_shape=orig_shape,
                                          strides=orig_strides,
                                          padding=orig_padding)
        elif operation.type == 'MaxPool':
            orig_strides = operation.get_attr('strides')
            return self.unpool(prev_tensor, mask, orig_strides)
            pass
        else:
            exit(
                'ERROR: You did not specify a reverse operation for type= ' + operation.type)

    def build_reverse_chain(self):
        print('\nBuilding reverse chain')

        last_tensor_shape = self.remembered_tensors_list[-1][0].shape
        last_tensor = tf.placeholder(tf.float32, last_tensor_shape)
        for i in reversed(self.remembered_tensors_list):
            i[2] = last_tensor
            last_tensor = self.reverse_operation(i[0], i[1], i[2], i[3])

        self.input_viz = last_tensor
        print('Building done\n')
        return last_tensor

    def unpool(self, net, mask, strides):
        """
          https: // github.com / yselivonchyk / Tensorflow_WhatWhereAutoencoder / blob / master / WhatWhereAutoencoder.py
        """
        with tf.name_scope('UnPool2D'):
            ksize = strides
            input_shape = net.get_shape().as_list()
            input_shape[0] = self.batch_size
            output_shape = (
                input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2],
                input_shape[3])
            print('ksize=' + str(ksize))
            print('input_shape=' + str(input_shape))
            print('output_shape=' + str(output_shape))
            # calculation indices for batch, height, width and feature maps
            one_like_mask = tf.ones_like(mask)
            batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64),
                                     shape=[input_shape[0], 1, 1, 1])
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

    def viz_channel(self, sess, tensor_name, tensor, reverse_tensor, ch_index,
                    channel_top_indices):
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
        # for simplicity, to avoid variable size batch:
        # len(channel_top_indices) = batch_size
        assert (len(channel_top_indices) == self.batch_size)

        activation_data = sess.run(tensor, feed_dict={
            self.input_ph: self.test_images.get_batch(channel_top_indices)})

        print('\tLayer=' + str(tensor))
        print('\tVisualizing channel= ' + str(ch_index))
        print('\t\tTop indices=' + str(channel_top_indices[:self.viz_matrix_size]))
        print('\t\tnp.max(activation_data) for all image on channel_' + str(
            ch_index) + ' =' + str(
            np.max(activation_data[0, :, :, ch_index])))
        print('\t\tnp.min(activation_data) for all image on channel_' + str(
            ch_index) + ' =' + str(
            np.min(activation_data[0, :, :, ch_index])))

        if False:
            for j in xrange(self.viz_matrix_size):
                t = activation_data[j, :, :, ch_index]
                print(j, channel_top_indices[j], np.sum(t), np.max(t), np.min(t))

        selected_activation_data = np.zeros(activation_data.shape, dtype=np.float32)
        if ch_index is not None:
            selected_activation_data[:, :, :, ch_index] = activation_data[:, :, :,
                                                          ch_index]
        else:
            selected_activation_data = activation_data
        input_viz_to_save = sess.run(self.input_viz,
                                     feed_dict={reverse_tensor: selected_activation_data,
                                                self.input_ph: self.test_images.get_batch(
                                                    channel_top_indices)})

        input_image_shape = self.input_ph.shape.as_list()
        input_viz_shape = list(input_viz_to_save.shape)
        assert (input_image_shape[1] == input_viz_shape[1])
        assert (input_image_shape[2] == input_viz_shape[2])

        # switch to viz_matrix_size
        #   - channel_top_indices
        #   - orig images
        #   - input viz
        #   - selected_activation_data
        input_viz_shape[0] = self.viz_matrix_size
        channel_top_indices = channel_top_indices[:self.viz_matrix_size]
        orig_images = self.test_images.get_batch(channel_top_indices)
        input_viz_to_save = input_viz_to_save[:self.viz_matrix_size, :, :, :]
        selected_activation_data = selected_activation_data[
                                   :self.viz_matrix_size, :, :, :]

        # activation image is greyscale, but not in 0...255
        # i normalize it (normalization is inside the prep function)
        # todo: move outside for clearity
        p_tensor_data, p_xys = self.prep_for_save_activation_image(
            selected_activation_data,
            ch_index,
            input_viz_shape)

        # orig images is not normalized
        # test_images shall implement a proper save function
        p_orig_images, p_orig_images_h \
            = self.prep_for_save_orig_image(orig_images,
                                            ch_index,
                                            input_viz_shape,
                                            p_xys,
                                            self.remembered_reception_sizes[
                                                tensor.name])
        # vizualization is not in 0...255
        # actually it could vary quite much:
        # I normalize each image in the batch
        p_viz_images, p_viz_images_h = self.prep_for_save_orig_image(
            image_norm4(input_viz_to_save),
            ch_index,
            input_viz_shape, p_xys,
            self.remembered_reception_sizes[tensor.name])

        self.test_images.save_pic(tensor_name + '/' + str(ch_index) + 'a', p_tensor_data)
        self.test_images.save_pic(tensor_name + '/' + str(ch_index) + 'o', p_orig_images)
        self.test_images.save_pic(tensor_name + '/' + str(ch_index) + 'oh',
                                  p_orig_images_h)
        self.test_images.save_pic(tensor_name + '/' + str(ch_index) + 'v', p_viz_images)
        self.test_images.save_pic(tensor_name + '/' + str(ch_index) + 'vh',
                                  p_viz_images_h)

    def prep_for_save_activation_image(self, tensor_data, ch_index, out_shape):
        print('\t\t prep_for_save_activation_image>')
        td_shape = tensor_data.shape
        tile_size = int(np.ceil(np.sqrt(out_shape[0])))
        assert (out_shape[0] == td_shape[0])
        print('\t\t out_shape=' + str(out_shape))
        print('\t\t td_shape=' + str(td_shape))
        input_ratio = float(out_shape[1]) / td_shape[1]
        print('\t\t zooming to activation by factor=' + str(input_ratio))

        if ch_index is None:
            return make_tiles(np.ones(out_shape) * 255, tile_size), None

        channel_data = tensor_data[:, :, :, ch_index]
        xys = self.max_activation_index_for_batch(channel_data)
        p_xys = self.scale_indices_to_input_space(xys, input_ratio=input_ratio)

        if True:
            for j in xrange(td_shape[0]):
                print("\t\t max activation at layer          =", j, "is at", xys[j])
                print("\t\t max activation projected to image=", j, "is at", p_xys[j])

        new_channel_data = ndimage.zoom(channel_data, (1, input_ratio, input_ratio),
                                        order=0)

        sh = new_channel_data.shape
        new_channel_data = image_norm4(new_channel_data.reshape([sh[0], sh[1], sh[2], 1]))

        return make_tiles(new_channel_data, tile_size), p_xys

    def prep_for_save_orig_image(self, tensor_data, ch_index, out_shape, p_xys,
                                 reception_size):
        print('\t\t prep_for_save_orig_image>')
        print('\t\t reception size= ' + str(reception_size))
        td_shape = tensor_data.shape
        tile_size = int(np.ceil(np.sqrt(out_shape[0])))
        assert (out_shape[0] == td_shape[0])

        if p_xys is not None:
            p_xys_short = [(i, j) for (i, j, k) in p_xys]

            new_tensor_data = center_on(tensor_data, p_xys_short, reception_size)
            red_tensor_data = color_on(tensor_data, p_xys_short, reception_size,
                                       pixel_value=255, color_dim=2)
            input_ratio = out_shape[1] / new_tensor_data.shape[1]
            print('\t\t\tinput_ratio', input_ratio)
            new_channel_data = ndimage.zoom(new_tensor_data,
                                            (1, input_ratio, input_ratio, 1), order=0)
            print('\t\t\tnew_tensor_data.shape', new_tensor_data.shape)
            print('\t\t\tnew_channel_data.shape', new_channel_data.shape)
        else:
            new_channel_data = tensor_data
            red_tensor_data = tensor_data

        return make_tiles(new_channel_data, tile_size), make_tiles(red_tensor_data,
                                                                   tile_size)

    @staticmethod
    def max_activation_index_for_batch(pics2D_batch):
        ret = []
        for i in xrange(pics2D_batch.shape[0]):
            (x, y, m) = max_in_2D(pics2D_batch[i, :, :])
            # ret.append((x * input_ratio, y * input_ratio, m))
            ret.append((x, y, m))
        return ret

    @staticmethod
    def scale_indices_to_input_space(xys, input_ratio=1):
        ret = []
        for (x, y, m) in xys:
            ret.append((int(x * input_ratio), int(y * input_ratio), m))
        return ret

    def viz(self, sess, tensor_name, mode='sum', viz_top=9):
        if self.input_viz is None:
            print("self.input_viz is None. Have you run build_reverse_chain()?")
            exit()
        success = False
        for i in self.remembered_tensors_list:
            tensor = i[0]
            operation = i[1]
            reverse_tensor = i[2]
            num_of_channels = tensor.shape[3]

            # limit num of channels:
            num_of_channels = np.min((num_of_channels, self.max_channel_num))

            if tensor.name == tensor_name:
                success = True

                top_indices = self.max_activation_for_layers(sess, tensor, mode=mode)

                tensor_name_unix = self.tensor_pretty_name(tensor_name) + mode
                print(
                    'Visualizing tensor' + tensor_name + '\t in folder=' + tensor_name_unix + '\tits shape=' + str(
                        tensor.shape) + " Activation calc mode=" + mode)
                target = self.target_dir + '/' + tensor_name_unix
                self.start_clean_dir(target)

                for channel_index in xrange(num_of_channels):
                    if mode == 'fix':
                        channel_top_indices = list(xrange(self.batch_size))
                    else:
                        channel_top_indices = top_indices[channel_index, :]
                    self.viz_channel(sess, target, tensor, reverse_tensor, channel_index,
                                     channel_top_indices)
                self.viz_channel(sess, target, tensor, reverse_tensor, None,
                                 list(xrange(128)))
        if not success:
            print('Error')
            print('Not found=' + tensor_name)
            print('In')
            for i in self.remembered_tensors_list:
                tensor = i[0]
                print('\t' + tensor.name)

    def max_activation_for_layers(self, sess, layer, mode='sum'):
        test_size = self.test_images.len()
        print('Searching max activation through the test dataset, test_size={}'.format(
            test_size))
        total_batch = int(test_size / self.batch_size)
        num_of_images = total_batch * self.batch_size
        num_of_channels = layer.shape[3]
        sum_activation_data = np.zeros((num_of_images, num_of_channels), dtype=np.float32)
        max_activation_data = np.zeros((num_of_images, num_of_channels), dtype=np.float32)

        for i in xrange(total_batch):
            batch_start = i * self.batch_size
            batch_end = (i + 1) * self.batch_size
            batch = self.test_images.get_batch(xrange(batch_start, batch_end))
            tensor_data = sess.run(layer, feed_dict={self.input_ph: batch})
            sum_activation_data[batch_start:batch_end, :] = np.sum(tensor_data,
                                                                   axis=(1, 2))
            max_activation_data[batch_start:batch_end, :] = np.max(tensor_data,
                                                                   axis=(1, 2))

        print(str(layer) + ' sum_activation_data.shape=' + str(sum_activation_data.shape))

        top_indices = np.zeros((num_of_images, self.batch_size), dtype=np.int32)
        for ch in xrange(num_of_channels):
            activation_base = sum_activation_data
            if mode == 'max':
                activation_base = max_activation_data
            top_indices[ch, :] = sorted(range(num_of_images),
                                        key=lambda i: activation_base[i, ch],
                                        reverse=True)[:self.batch_size]

        if False:
            # for ch in xrange(num_of_channels):
            for ch in xrange(1):
                batch_indices = top_indices[ch, :]
                batch = self.test_images.get_batch([batch_indices])
                tensor_data = sess.run(layer, feed_dict={self.input_ph: batch})
                for j in xrange(self.batch_size):
                    t = tensor_data[j, :, :, ch]
                    print(j, batch_indices[j], np.sum(t), np.max(t), np.min(t))

        return top_indices

    def tensor_pretty_name(self, tensor_name):
        return tensor_name.replace('/', "").replace(':', "")

    def start_clean_dir(self, target):
        if os.path.exists(target):
            shutil.rmtree(target)
        if not os.path.exists(target):
            os.makedirs(target)