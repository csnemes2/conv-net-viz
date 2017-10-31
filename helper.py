from scipy.misc import imread, imsave
from PIL import Image
import numpy as np


def load_pic2(img_path):
    im1 = (imread(img_path)[:, :, :3]).astype(np.float32)
    im1 = im1 - np.mean(im1)
    # im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]
    im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2].copy(), im1[:, :, 0].copy()
    return im1


def load_pic(img_path):
    im1 = np.array(Image.open(img_path, mode='r').convert('RGB'), dtype=np.float32)
    im1 = im1 - np.mean(im1)
    im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2].copy(), im1[:, :, 0].copy()
    return im1


def save_pic_old(img_path, rgb):
    imsave(img_path, rgb)


def save_pic(img_path, rgb):
    if len(rgb.shape) == 2:
        img = Image.fromarray(np.uint8(rgb))
    elif len(rgb.shape) == 3:
        if rgb.shape[2] == 1:
            img = Image.fromarray(np.uint8(rgb[:, :, 0]))
        else:
            img = Image.fromarray(np.uint8(rgb), 'RGB')
    img.save(img_path + '.jpg', quality=100)


def pop_file(img_path):
    I = np.asarray(Image.open(img_path))
    im = Image.fromarray(np.uint8(I))
    im.show()


def pop_pic(im1, title=''):
    im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2].copy(), im1[:, :, 0].copy()
    img = Image.fromarray(np.uint8(im1), 'RGB')
    img.show()


def image_norm4(data):
    for i in xrange(data.shape[0]):
        amin = np.min(data[i, :, :, :])
        positive = data[i, :, :, :] + np.abs(amin)
        amax = np.max(positive)
        data[i, :, :, :] = positive / amax * 255
    return data


def make_tiles(data, tile_size):
    sh = list(data.shape)
    # print (sh)
    # +1: white padding on the right of each cell
    ret_shape = [(sh[1] + 1) * tile_size, (sh[2] + 1) * tile_size, sh[3]]
    ret = np.ones(ret_shape) * 255
    idx = 0
    for i in xrange(tile_size):
        start_row = i * (sh[1] + 1)
        for j in xrange(tile_size):
            start_col = j * (sh[2] + 1)
            if idx < sh[0]:
                # print(start_row, start_col)
                ret[start_row:start_row + sh[1], start_col:start_col + sh[2], :] = data[
                                                                                   idx, :,
                                                                                   :, :]
                idx += 1
    return ret

def max_in_2D(pics2D):
    x, y = np.unravel_index(np.argmax(pics2D), pics2D.shape)
    m = pics2D[x, y]
    return (x, y, m)

def convert_4D_RGB(img):
    assert (len(img.shape) == 4)

    if img.shape[3] == 1:
        #print ('\t\t\t == converting image to RGB ==')
        new_shape = [img.shape[0], img.shape[1], img.shape[2], 3]
        new_img = np.zeros(new_shape)
        new_img[:, :, :, 0] = img[:, :, :, 0]
        new_img[:, :, :, 1] = img[:, :, :, 0]
        new_img[:, :, :, 2] = img[:, :, :, 0]
        return new_img

    return img
