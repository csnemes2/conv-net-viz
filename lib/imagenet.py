import glob
import io
import numpy as np

class ImageDB:
    def __init__(self, imagenet_dir):
        self.imagenet_dir = imagenet_dir
        print("Loading imagenet filenames")
        self.image_names = glob.glob(self.imagenet_dir + "/*.JPEG")
        self.num_of_images = len(self.image_names)
        print("Done. Len="+str(self.num_of_images))
        self.image_shape = io.load_pic_BGR(self.image_names[0]).shape
        print("Image shape=" + str(self.image_shape))

    def limit_len(self, new_len):
        self.num_of_images = new_len

    def len(self):
        return self.num_of_images

    # Loading images BGR
    # pixel values 0...255
    def get_batch(self, indices):
        sh=[len(indices)]+list(self.image_shape)
        ret = np.zeros(sh)
        if len(ret) == 0:
            print ('Warning empty batch requested, really?')
        if np.max(indices)>=self.num_of_images:
            print ('Wrong index:{}'.format(np.max(indices)))
            exit()
        counter=0
        for i in indices:
            ret[counter,:,:,:]=io.load_pic_BGR(self.image_names[i])
            counter+=1
        return ret

    # Reversing BGR to RGB
    def save_pic(self, img_path, rgb):
        io.save_pic_BGR(img_path, rgb)
