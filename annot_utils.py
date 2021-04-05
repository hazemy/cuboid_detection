#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
from PIL import Image 
    
    
def convert2jpg(images_dir):
    '''
    Converts images from png to jpg. Transparent png images are converted to jpg with white background
    '''
    for filename in os.listdir(images_dir):
         if filename.endswith(".png"): 
             im_path = os.path.join(images_dir, filename)
             im = Image.open(im_path)
             im.load()
             if im.mode != 'RGBA': # 'A' for alpha channel that encodes transparency
                 im = im.convert('RGBA')
             background = Image.new("RGB", im.size, (255, 255, 255))
             background.paste(im, mask=im.split()[3]) # 3 is the alpha channel
             mod_im_path = os.path.join(images_dir, filename.split('.')[-2] + '.jpg')
             background.save(mod_im_path, 'JPEG', quality=80)
             os.remove(im_path)
                 


if __name__ == '__main__':
    images_dir = '/home/porthos/masters_thesis/datasets/augmented_dataset/images' 
    convert2jpg(images_dir)