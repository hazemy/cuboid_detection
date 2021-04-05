#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# from mat2cuboid_annot import mat2cuboid_annot
# from get_2d_bbox import get_bbox_abs, get_bbox_xywh
import os
import cv2
import json
import numpy as np
from detectron2.structures import BoxMode
from annot_processor import merge_annot_files, remove_faulty, check_visibilty


def convert2detectron_format(annot_list, images_dir):
    '''
    inputs: annotations list to be converted
            path to 'images' folder in the dataset - used for retreiving fileName
    ouput: standard detectron 2 dataset format (json_like)
    It is required that all images are in an 'images' folder in the same 
    directory as the annotation file
    
    '''    
    # with open(annot_file) as f:
    #     annot_list = json.load(f)
        
    dataset_list = []
    for annot in annot_list: #per image loop
        record = {}       
        entry = annot
        
        image_id = entry['fileName'].split('/')[-1]
        image_id = image_id.split('\\')[-1]
        image_id = image_id.split('.')[-2] #use last section of image directory as image id
        record['image_id'] = image_id
        # print(image_id)
        
        # images_dir = annot_file.split('/')[:-1]
        # img_path = os.path.join('/', *images_dir, 'images', image_id + '.jpg') #construct path to images folder. * used to force accepting lists
        # images_dir = images_dir.split('/')
        img_path = os.path.join(images_dir, image_id + '.jpg') #construct path to images folder. * used to force accepting lists
        record['file_name'] = img_path
        
        height, width = cv2.imread(img_path).shape[:2]
        record['height'] = height
        record['width'] = width

        annotations = []
        for i in range(len(entry['squares'])): #per object instance loop
            instance = {}
            bbox = entry['squares'][i]           
            if bbox: #non-empty annotations
                instance['category_id'] = 0 #foreground (cuboid)
            else:
                instance['category_id'] = 1 #background 
                                             #ToDo: leave category empty for background?              
            try:
                bbox_abs = get_bbox_abs_coor(bbox, height, width)
                bbox_xywh = get_bbox_xywh(bbox_abs)
            except IndexError:
                print('2D BBox could not be retreived!')
                return
            
            cuboid = entry['cubes'][i]
            if cuboid: #non-empty
                cuboid_abs = get_cuboid_abs_coor(cuboid, height, width)
                # cuboid_interleaved = interleave_visibilty(cuboid_abs)
            
            instance['bbox'] = bbox_xywh
            instance['bbox_mode'] = BoxMode.XYWH_ABS 
            # instance['keypoints'] = cuboid_interleaved
            instance['keypoints'] = cuboid_abs
            annotations.append(instance)
        record['annotations'] = annotations   
        dataset_list.append(record)
        
    # with open("test.json", "w") as write_file:
    #     json.dump(dataset_list, write_file, indent=4, sort_keys=True) 
        
    return dataset_list

def get_bbox_abs_coor(coordinates, height, width):
    '''
    converts coordinates of the corners of a single cuboid or bbox to absolute pixel positions
    '''
    coors_abs = [] #all converted box coordinates
    for coors in coordinates: #convert each x and y coordinate
        conv_coors = [] #single converted x,y coordinates
        coor_abs_x = coors[0]*width
        conv_coors.append(coor_abs_x)
        coor_abs_y = coors[1]*height
        conv_coors.append(coor_abs_y) 
        coors_abs.append(conv_coors)
    return coors_abs

def get_cuboid_abs_coor(coordinates, height, width):
    '''
    converts coordinates of the corners of a single cuboid or bbox to absolute pixel positions.
    Also, converts a list[list] containing x,y corrdinates of the corners of the cuboid into a single list
    '''
    # coors_abs = [] #all converted box coordinates
    cuboid_mod = []
    for coors in coordinates: #convert each x and y coordinate
        conv_coors = [] #single converted x,y coordinates
        coor_abs_x = coors[0]*width
        conv_coors.append(coor_abs_x)
        coor_abs_y = coors[1]*height
        conv_coors.append(coor_abs_y)
        conv_coors.append(coors[2]) #adds visibility
        # coors_abs.append(conv_coors)
        cuboid_mod.extend(conv_coors)
    return cuboid_mod

def get_bbox_xywh (bbox_abs): #(x,y) is that of the top-left corner
    '''
    returns a single list containing the xywh bbox conversion
    '''  
    x_min = bbox_abs[0][0] #hanndle 1 box only
    x_max = bbox_abs[2][0]
    y_min = bbox_abs[0][1]
    y_max = bbox_abs[2][1]  
    w = x_max - x_min
    h = y_max - y_min
    bbox_xywh = [x_min, y_min, w, h] #converted bbox
    return bbox_xywh

# def interleave_visibilty(cuboid, dummy):
#     '''
#     Input: - a list[list] containing x,y corrdinates of the corners of the cuboid
#            - a flag for interleaving with dummy visibility values. Set to False to use actual values in annotation
#     Output: a single list of x,y corner coordinates of the cuboid interleaved with visibilty value
#     '''
#     v = 2 #visibility
#     cuboid_mod = []
#     # cuboid_mod.extend(cuboid_coors)
#     # cuboid_mod.append(v)
#     for i in range(len(cuboid)):
#         cuboid_mod.extend(cuboid[i])
#         cuboid_mod.append(v)
            
#     return cuboid_mod     
    

if __name__ == '__main__':
    annot_file_dir = '/home/porthos/masters_thesis/datasets/augmented_dataset/annotations_hazem.json'
    annot_files_dir_list = [annot_file_dir]
    annot_files_merged = merge_annot_files(annot_files_dir_list)
    annot_file_corrected = remove_faulty(annot_files_merged)
    check_visibilty(annot_file_corrected)
    images_dir = '/home/porthos/masters_thesis/datasets/augmented_dataset/images'
    dataset_list = convert2detectron_format(annot_file_corrected, images_dir)
    
    
    
    
    
    
    
    