#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 18:40:29 2020

@author: porthos
"""

import os
import json
    

dataset_dir = '/home/porthos/masters_thesis/datasets/data_release/data_release/cuboid'
json_file = os.path.join(dataset_dir, 'state.json')

with open(json_file) as f:
    bbox_annot_list = json.load(f)

def get_2d_bbox (cuboid_image_id, bbox_idx): 
    for i, entry in enumerate(bbox_annot_list):
        # image_id = re.split('\ | /| ' ,file_name_mod)[-1]
        bb_image_id = entry['fileName'].split('/')[-1]
        bb_image_id = bb_image_id.split('\\')[-1]
        bb_image_id = bb_image_id.split('.')[-2]
        print(bb_image_id)
        if (bb_image_id == cuboid_image_id):
            bbox = bbox_annot_list[i]['squares']
            print('Found at index {}'.format(i))
            selected_bbox = bbox[bbox_idx]
            bbox_mod = selected_bbox[0:2] #ToDo: extract annotation according to box mode
            return bbox_mod
        else:
            print('Image_id not found!')
            raise IndexError



if __name__ == '__main__':
    cuboid_image_id = '000004'
    get_2d_bbox(cuboid_image_id)
    