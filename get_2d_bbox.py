#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 18:40:29 2020

@author: porthos
"""

import os
import json
import cv2

    

# dataset_dir = '/home/porthos/masters_thesis/datasets/data_release/data_release/cuboid'
# json_file = os.path.join(dataset_dir, 'state.json')

# with open(json_file) as f:
#     bbox_annot_list = json.load(f)

# def get_2d_bbox (cuboid_image_id, height, width): #supports images with 1 bbox for now
#     for i, entry in enumerate(bbox_annot_list): #search with image_id
#         # image_id = re.split('\ | /| ' ,file_name_mod)[-1]
#         bb_image_id = entry['fileName'].split('/')[-1]
#         bb_image_id = bb_image_id.split('\\')[-1]
#         bb_image_id = bb_image_id.split('.')[-2]
#         #print(bb_image_id)
#         if (bb_image_id == cuboid_image_id):
#             bbox = bbox_annot_list[i]['squares']
#             # bbox_scales = bbox_annot_list[i]['scale']
#             # print('Found at index {}'.format(i))
#             # selected_bbox = bbox[bbox_idx]
#             bbox_abs = get_bbox_abs(bbox, height, width) #ToDo: extract annotation according to box mode
#             bbox_xywh = get_bbox_xywh(bbox_abs) 
#             return bbox_xywh
#     print('Image_id not found!')
#     raise IndexError


def get_bbox_abs(bbox, height, width):
    '''
    converts coordinates of a single bbox to absolute pixel positions
    '''
    #bbox_abs = [] #all converted instances
    # for instance in bbox: #loop over each bbox instance --> [[[x1,y1], [x2,y2],...]]
    bbox_abs = [] #all converted box coordinates
    for coors in bbox: #convert each x and y coordinate
        bbox_coors = [] #single converted x,y coordinates
        coor_abs_x = coors[0]*width
        bbox_coors.append(coor_abs_x)
        coor_abs_y = coors[1]*height
        bbox_coors.append(coor_abs_y)
        bbox_abs.append(bbox_coors)
        # bbox_abs.append(bbox_coors)
    # print(bbox_abs)
    return bbox_abs
 

def get_bbox_xywh (bbox_abs): #(x,y) is that of the top-left corner
    '''
    returns a single list containing the xywh bbox conversion
    '''
    # bbox_xywh = [] #all converted box coordinates
    # for instance in bbox_abs: #loop over each bbox instance --> [[[x1,y1], [x2,y2],...]]
    # x_min = instance[0][0]
    # x_max = instance[2][0]
    # y_min = instance[0][1]
    # y_max = instance[2][1]
    
    # w = x_max - x_min
    # h = y_max - y_min
    
    x_min = bbox_abs[0][0] #hanndle 1 box only
    x_max = bbox_abs[2][0]
    y_min = bbox_abs[0][1]
    y_max = bbox_abs[2][1]
    
    w = x_max - x_min
    h = y_max - y_min
    
    # bbox_conv = [x_min, y_min, w, h] #converted bbox
    bbox_xywh = [x_min, y_min, w, h] #converted bbox
    # bbox_xywh.append(bbox_conv)
    # print(bbox_xywh)
    return bbox_xywh





if __name__ == '__main__':
    cuboid_image_id = '000004'
    img_dir = '/home/porthos/masters_thesis/datasets/partial_dataset/state_partial.json'
    height, width = cv2.imread(img_dir).shape[:2]
    # bbox = get_2d_bbox(cuboid_image_id, height, width)
    # print('The bbox is: {}'.format(bbox))
    
    # draw_bbox(img_dir, bbox)

    
    
    
    
    
    
    