#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 04:22:26 2020

@author: porthos
"""

from detectron2.data import DatasetCatalog, MetadataCatalog
# from mat2cuboid_annot import mat2cuboid_annot
from cuboid_annot2detectron_format import cuboid_annot2detectron_format
# import os
import cv2
import random
from detectron2.utils.visualizer import Visualizer




# def register_dataset(dataset_name, func):
#     '''
#     inputs: -dataset in detectron 2 standard format (list of dicts)
#             -desired name to register dataset with (can be different from original dataset name)
#     '''
#     DatasetCatalog.register(dataset_name + '_train', lambda: cuboid_annot2detectron_format(dataset_dir + '/train'))

    

# if __name__ == '__main__':
dataset_dir = '/home/porthos/masters_thesis/datasets/data_release/data_release/cuboid'
# cuboid_annot = mat2cuboid_annot(os.path.join(dataset_dir, 'Annotations.mat')) 
# dataset_list = cuboid_annot2detectron_format(cuboid_annot, dataset_dir)
# for d in ['train']: #ToDo: add validation
# DatasetCatalog.register('cuboid_dataset' + '_train', lambda: cuboid_annot2detectron_format(dataset_dir + '/train'))
DatasetCatalog.register('cuboid_dataset' + '_train', lambda: cuboid_annot2detectron_format(dataset_dir))
MetadataCatalog.get('cuboid_dataset' + '_train').set(thing_classes=['cuboid'])    
cuboid_metadata = MetadataCatalog.get("cuboid_train")
   
    
# DatasetCatalog.register('train', lambda: get_kitti_data_dict(dataset_root_dir, 'train'))
    
#     for d in ['train']:
# DatasetCatalog.register(d, lambda d: d = get_kitti_data_dict(dataset_root_dir, d))
# MetadataCatalog.get(d).thing_classes = ["person"]


dataset_dicts = cuboid_annot2detectron_format(dataset_dir)
# sample = random.sample(dataset_dicts, 3)
i = 0
for d in dataset_dicts[:4]:
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=cuboid_metadata, scale=1)
    out = visualizer.draw_dataset_dict(d)
    # out = visualizer.draw_instance_predictions(dataset_dicts["annotations"].to("cpu"))
    final_img = cv2.resize(out.get_image()[:, :, ::-1], (500,500))
    cv2.imshow('2D BB' + str(i), final_img)
    i=i+1
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    