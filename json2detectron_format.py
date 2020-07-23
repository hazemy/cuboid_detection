#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 17:51:02 2020

@author: porthos
"""


# from mat2cuboid_annot import mat2cuboid_annot
from get_2d_bbox import get_bbox_abs, get_bbox_xywh
import os
import cv2
import json
import numpy as np
from detectron2.structures import BoxMode
from annot_processor import merge_annot_files


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
        
        # images_dir = annot_file.split('/')[:-1]
        # img_path = os.path.join('/', *images_dir, 'images', image_id + '.jpg') #construct path to images folder. * used to force accepting lists
        # images_dir = images_dir.split('/')
        img_path = os.path.join(images_dir, image_id + '.jpg') #construct path to images folder. * used to force accepting lists
        record['file_name'] = img_path
        
        height, width = cv2.imread(img_path).shape[:2]
        record['height'] = height
        record['width'] = width

        annotations = []
        # bbox_idx = 0 #to be used for retrieving multiple instances
        # for j in range(len(entry['squares'])): #per object instance loop
        for bbox in entry['squares']: #per object instance loop
            instance = {}
            keypoints = []
            # vertices = entry['object'][j]['position'] #x,y coordinates of vertices
            # v_list = np.ones(8) #visibility of vertices - entry needed by detectron keypoints format
            # keypoints = [val for pair in zip(vertices[0], vertices[1]) for val in pair]  #interleaves the x,y coordinates in one list
             
            #ToDo: interleave coordinate pair with visibility value
            # keypoints = np.zeros(len(pairs)+len(ones)) #x,y coor and v for each vertex                                    
            # for k in range(len(pairs)):
            #     keypoints = np.zeros(24) #x,y coor and v for each vertex
            #     keypoints.insert(k+2, v_list[k])
    
            # instance['keypoints'] = keypoints
            
            if bbox: #non-empty annotations
                instance['category_id'] = 0 #foreground (cuboid)
            else:
                instance['category_id'] = 1 #background 
                                             #ToDo: leave category empty for background?   
            
            try:
                bbox_abs = get_bbox_abs(bbox, height, width)
                bbox_xywh = get_bbox_xywh(bbox_abs)
            except IndexError:
                print('2D BBox could not be retreived!')
                return

            instance['bbox'] = bbox_xywh #ToDo: why in floating pt. ?
            instance['bbox_mode'] = BoxMode.XYWH_ABS 
            annotations.append(instance)
        record['annotations'] = annotations   
        dataset_list.append(record)
        
    # with open("test.json", "w") as write_file:
    #     json.dump(dataset_list, write_file, indent=4, sort_keys=True) 
        
    return dataset_list
    
    

if __name__ == '__main__':
    # annot_file = '/home/porthos/masters_thesis/datasets/partial_dataset/state_partial.json'
    # cuboid_annot = mat2cuboid_annot(os.path.join(dataset_dir, 'Annotations.mat')) 
    annot_files_dir_list = []
    annot_file_dir_1 = '/home/porthos/masters_thesis/datasets/full_dataset/state_hazem.json'
    annot_file_dir_2 = '/home/porthos/masters_thesis/datasets/full_dataset/state_mojtaba.json'   
    annot_file_dir_3= '/home/porthos/masters_thesis/datasets/full_dataset/state_frederick.json'   
    annot_file_dir_4 = '/home/porthos/masters_thesis/datasets/full_dataset/state_ammar.json'   
    annot_files_dir_list = [annot_file_dir_1, annot_file_dir_2, annot_file_dir_3, annot_file_dir_4]
    annot_files_merged = merge_annot_files(annot_files_dir_list)
    images_dir = '/home/porthos/masters_thesis/datasets/full_dataset/images'
    dataset_list = convert2detectron_format(annot_files_merged, images_dir)
    
    
    
    
    
    
    
    