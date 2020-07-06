#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 17:51:02 2020

@author: porthos
"""


from mat2cuboid_annot import mat2cuboid_annot
from get_2d_bbox import get_2d_bbox
import os
import cv2
import json
import numpy as np
from detectron2.structures import BoxMode


def cuboid_annot2detectron_format(dataset_dir):
    '''
    input: dataset directory containing all images (needed in order to append directory of filename path)
    ouput: standard detectron 2 dataset format (json_like)
    
    '''
    # cuboid_annot = mat2cuboid_annot('Annotations.mat')
    cubiod_annot = cuboid_annot = mat2cuboid_annot(os.path.join(dataset_dir, 'Annotations.mat')) 
    # dataset_dir = '/home/porthos/masters_thesis/datasets/data_release/data_release/cuboid'
    dataset_list = []
    for i in range(len(cuboid_annot)): #per image loop
        '''
        converts the intermediate format (cuboid annotation) to the standard dataset format for 
        Detectron 2, as by: https://detectron2.readthedocs.io/tutorials/datasets.html
        '''
        record = {}
        
        entry = cuboid_annot[i]
        img_path = os.path.join(dataset_dir, entry['image'])
        # record['filename'] = os.path.join(dataset_dir, img_path)
        record['file_name'] = img_path

        
        height, width = cv2.imread(img_path).shape[:2]
        record['height'] = height
        record['width'] = width
        
        image_id = record['file_name'].split('/')[-1] 
                                                  #use last section of image directory as image id
        record['image_id'] = image_id
        
        annotations = []
        bbox_idx = 0
        for j in range(len(entry['object'])): #per object instance loop
            instance = {}
            keypoints = []
            vertices = entry['object'][j]['position'] #x,y coordinates of vertices
            # v_list = np.ones(8) #visibility of vertices - entry needed by detectron keypoints format
            keypoints = [val for pair in zip(vertices[0], vertices[1]) for val in pair]  #interleaves the x,y coordinates in one list
             
            #ToDo: interleave coordinate pair with visibility value
            # keypoints = np.zeros(len(pairs)+len(ones)) #x,y coor and v for each vertex                                    
            # for k in range(len(pairs)):
            #     keypoints = np.zeros(24) #x,y coor and v for each vertex
            #     keypoints.insert(k+2, v_list[k])
    
            #instance['keypoints'] = keypoints
            if (entry['object'][j]['type'] == 'cuboid'):
                instance['category_id'] = 1 #foreground (cuboid)
            else:
                instance['category_id'] = 2 #background 
                                             #ToDo: leave category empty for background?   
            
            # bbox = list(np.zeros(4))
            try:
                bbox = get_2d_bbox(image_id, bbox_idx) #index of bbox to be returned
                bbox_idx = bbox_idx + 1
            except IndexError:
                print('2D BBox could not be retreived!')
                return

            instance['bbox'] = bbox
            instance['bbox_mode'] = BoxMode.XYXY_ABS #ToDo: why in floating pt. ?
            annotations.append(instance)
        record['annotations'] = annotations   
        dataset_list.append(record)
        
    # with open("test.json", "w") as write_file:
    #     json.dump(dataset_list, write_file, indent=4, sort_keys=True) 
        
    return dataset_list
    
    

if __name__ == '__main__':
    dataset_dir = '/home/porthos/masters_thesis/datasets/data_release/data_release/cuboid'
    # cuboid_annot = mat2cuboid_annot(os.path.join(dataset_dir, 'Annotations.mat')) 
    dataset_list = cuboid_annot2detectron_format(dataset_dir)
    
    
    
    
    
    
    
    