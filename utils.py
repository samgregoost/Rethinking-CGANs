import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import scipy.misc
from glob import glob
from PIL import Image
import tensorflow as tf

from sklearn.neighbors import NearestNeighbors

def to_rgb(images):
    stacked_img = np.stack((images,)*3, -1)
    stacked_img=np.squeeze(stacked_img)
    return stacked_img

def make_grid(img, gen_img, rows):
    gen_list = []
    stack = []
    for item in img:
        stack.append(item)
    gen_list.append(np.concatenate(stack,1))
    
    for row in np.split(gen_img, rows):
        stack = []
        for item in row:
            stack.append(np.squeeze(item))
        gen_list.append(np.concatenate(stack,1))
        
    return ((np.concatenate(gen_list,0)+1.0)*127.5).astype('uint8')

def normalize(x) :
    return x/127.5 - 1

def inverse_transform(images):
    return (images+1.)/2.

def exponential_moving_average(target_var,source_var,beta):
    ema_op =[]
    for indx in range(len(source_var)):
        ema_op.append(target_var[indx].assign(target_var[indx]*beta + source_var[indx]*(1.-beta)))
    return ema_op  

def copy_params(target_var,source_var):
    copy_op = []
    for indx in range(len(source_var)):
        copy_op.append(target_var[indx].assign(source_var[indx]))
    return copy_op
