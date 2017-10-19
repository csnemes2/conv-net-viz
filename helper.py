

from scipy.misc import imread,imsave
from PIL import Image
import numpy as np



def load_pic2(img_path):
    im1 = (imread(img_path)[:,:,:3]).astype(np.float32)
    im1 = im1 - np.mean(im1)
    #im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]
    im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2].copy(), im1[:, :, 0].copy()
    return im1

def load_pic(img_path):
    im1 = np.array(Image.open(img_path,mode='r').convert('RGB'),dtype=np.float32)
    im1 = im1 - np.mean(im1)
    im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2].copy(), im1[:, :, 0].copy()
    return im1

def save_pic_old(img_path,rgb):
    imsave(img_path, rgb)

def save_pic(img_path,rgb):
    img = Image.fromarray(rgb)
    img.convert('RGB').save(img_path+'.jpg')

def pop_file(img_path):
    I = np.asarray(Image.open(img_path))
    im = Image.fromarray(np.uint8(I))
    im.show()

def pop_pic(im1,title=''):
    im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2].copy(), im1[:, :, 0].copy()
    img = Image.fromarray(np.uint8(im1), 'RGB')
    img.show()