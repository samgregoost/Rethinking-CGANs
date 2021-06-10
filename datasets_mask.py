import glob
import random
import os
import numpy as np
from PIL import Image
import tensorflow as tf

class ImageDataset(object):
    def __init__(self, root, img_size=128, load_size=None, mask_size=64, mode='train', crop_mode='random'):
        self.img_size = img_size
        self.load_size = load_size
        self.mask_size = mask_size
        self.mode = mode
        self.files = sorted(glob.glob('%s/*.jpg' % root))
        self.files = self.files[:-4000] if mode == 'train' else self.files[-4000:]
        self.crop_mode=crop_mode

    def crop_and_resize(self,img):
        x,y = img.size
        ms = min(img.size)
        x_start = (x-ms)//2
        y_start = (y-ms)//2
        x_stop = x_start + ms
        y_stop = y_start + ms
        img = img.crop((x_start, y_start, x_stop, y_stop))
        img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
        return img

    def transform(self,img):
        return np.array(img,'float32')/ 127.5 -1
    
    def apply_random_mask(self, img):
        """Randomly masks image"""
        y1, x1 = np.random.randint(0, self.img_size-self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        mask = np.zeros((self.img_size, self.img_size, 1), 'float32')
        mask[x1:x2, y1:y2, 0] = 1
        masked_part = img.crop((x1, y1, x2, y2)).copy()
        masked_img = img.copy()
        for i in range(x1,x2):
            for j in range(y1,y2):
                masked_img.putpixel((i,j), (255,255,255))

        return masked_img, masked_part, mask

    def apply_center_mask(self, img):
        """Mask center part of image"""
        # Get upper-left pixel coordinate
        i = (self.img_size - self.mask_size) // 2
        mask = np.zeros((self.img_size, self.img_size, 1), 'float32')
        mask[i:i+self.mask_size, i:i+self.mask_size,0] = 1
        masked_part = img.crop((i, i, i+self.mask_size, i+self.mask_size))
        masked_img = img.copy()
        for j in range(i,i+self.mask_size):
            for k in range(i,i+self.mask_size):
                masked_img.putpixel((j,k), (255,255,255))

        return masked_img, masked_part, mask

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        img = self.crop_and_resize(img)
        #img = self.transform(img)
        if self.mode == 'train':
            if self.crop_mode=='random':
                # For training data perform random mask
                masked_img, aux, mask = self.apply_random_mask(img)
            elif self.crop_mode == 'none':
                masked_img, aux, mask = self.apply_center_mask(img)
        else:
            # For test data mask the center of the image
            masked_img, aux, mask = self.apply_center_mask(img)

        return self.transform(img), self.transform(masked_img), self.transform(aux), mask

    def __len__(self):
        return len(self.files)