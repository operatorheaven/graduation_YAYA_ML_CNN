from PIL import Image
import os,sys
from multiprocessing import Pool
import scipy.misc as misc
import shutil
import random
import numpy as np
import cv2 as cv

input_dir = '/Users/wangmingjian/All_code/python/YAYA/data_in/pic/coil-100/'
output_dir = '/Users/wangmingjian/All_code/python/YAYA/data_in/pic/blurred_coil-100_ver1.1/'
scale = 4.0
image_list = os.listdir(input_dir)
image_list = [os.path.join(input_dir, _) for _ in image_list]

def downscale(name):
    print(name)
    with Image.open(name) as im:
        w, h = im.size 
        w_new = int(w / scale) 
        h_new = int(h / scale) 
        im_new = im.resize((w_new, h_new), Image.ANTIALIAS) 
        save_name = os.path.join(output_dir, name.split('/')[-1]) 
        im_new.save(save_name)

def resize(image, newsize):
    #methods = ['nearest','lanczos','bilinear','bicubic','cubic']
    methods = ['bicubic','lanczos','bilinear','bicubic','cubic']
    scale = [4, 3, 2, 1.5, 0.75, 0.5, 0.25]
    #0.309090465205
    value = random.random()
    if (len(image.shape) == 3):
        w, h, d = image.shape
    else:
        w, h = image.shape 
    s = int(value*1000) % 7
    if s == 0:
        #image = image + np.random.random_sample(image.shape)*(int(value*100) % 50)
        image = cv.GaussianBlur(image, (21, 21), 0)
    elif s == 1:
        #image = cv.blur(image, (5, 5))
        image = cv.GaussianBlur(image, (23, 23), 0)
    elif s == 2:
        image = cv.GaussianBlur(image, (25, 25), 0)
    elif s == 3:
        #image = cv.medianBlur(image, 11)
        image = cv.GaussianBlur(image, (13, 13), 0)
    elif s == 4:
        #cv.bilateralFilter(image, 9, 75, 75)
        #image = cv.medianBlur(image, 13)
        image = cv.GaussianBlur(image, (15, 15), 0)
    elif s == 5:
        image = cv.GaussianBlur(image, (17, 17), 0)
    elif s == 6:
        image = cv.GaussianBlur(image, (19, 19), 0)
    image = misc.imresize(image, newsize, methods[int(value*10)%5])
    for i in range(int(value*100) % 5):
        image = misc.imresize(image, (int(newsize[0]*scale[int(value*(10 ** (1+i)))%7]), int(newsize[1]*scale[int(value*(10**(1+i)))%7])), methods[int(value*(10 ** (4+i)) % 5)])
        image = misc.imresize(image, newsize, methods[int(value*(10 ** (4+i)) %
                                                          10 - 5)])
    return image
counter = 0
for single_image in image_list:
    image = misc.imread(single_image)
    w, h, d = image.shape
    w_new = int(w/scale)
    h_new = int(h/scale)
    image = resize(image, (w_new, h_new))
    save_name = os.path.join(output_dir, '1.'+ single_image.split('/')[-1])
    misc.imsave(save_name, image)
    counter += 1
    print('\r>>Converting image ' + str(counter*100.0/len(image_list)) + ' %\n')
print('Convert Over!\n')
