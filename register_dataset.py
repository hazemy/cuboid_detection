#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 04:22:26 2020

@author: porthos
"""

from detectron2.data import DatasetCatalog, MetadataCatalog
# from mat2cuboid_annot import mat2cuboid_annot
# from cuboid_annot2detectron_format import cuboid_annot2detectron_format
from format_converter import convert2detectron_format
from dataset_splitter import split_dataset
from annot_processor import merge_annot_files, filter_duplicates



# annot_files_dir_list = []
# annot_file_dir_1 = '/home/porthos/masters_thesis/datasets/full_dataset/state_hazem.json'
# annot_file_dir_2 = '/home/porthos/masters_thesis/datasets/full_dataset/state_mojtaba.json'   
# annot_file_dir_3= '/home/porthos/masters_thesis/datasets/full_dataset/state_frederick.json'   
# annot_file_dir_4 = '/home/porthos/masters_thesis/datasets/full_dataset/state_ammar.json'   
# annot_files_dir_list = [annot_file_dir_1, annot_file_dir_2, annot_file_dir_3, annot_file_dir_4]
# annot_files_merged = merge_annot_files(annot_files_dir_list)

annot_file_dir = '/home/porthos/masters_thesis/datasets/full_dataset/state_all_revised.json'
images_dir = '/home/porthos/masters_thesis/datasets/full_dataset/images'
annot_files_dir_list = [annot_file_dir]
annot_files_merged = merge_annot_files(annot_files_dir_list)

annot_file_filtered = filter_duplicates(annot_files_merged)
dataset_list = convert2detectron_format(annot_file_filtered, images_dir)
train_dataset, val_dataset, test_dataset = split_dataset(dataset_list, 0.6, 0.2)
# DatasetCatalog.register('cuboid_dataset' + '_train', lambda: cuboid_annot2detectron_format(dataset_dir + '/train'))

DatasetCatalog.register('cuboid_dataset' + '_train', lambda: train_dataset)
MetadataCatalog.get('cuboid_dataset' + '_train').set(thing_classes=['cuboid'])    
cuboid_metadata_train = MetadataCatalog.get("cuboid_dataset_train")

DatasetCatalog.register('cuboid_dataset' + '_val', lambda: val_dataset)
MetadataCatalog.get('cuboid_dataset' + '_val').set(thing_classes=['cuboid'])    
cuboid_metadata_val = MetadataCatalog.get("cuboid_dataset_val")
  
DatasetCatalog.register('cuboid_dataset' + '_test', lambda: test_dataset)
MetadataCatalog.get('cuboid_dataset' + '_test').set(thing_classes=['cuboid'])    
cuboid_metadata_test = MetadataCatalog.get("cuboid_dataset_test") 
    
# DatasetCatalog.register('train', lambda: get_kitti_data_dict(dataset_root_dir, 'train'))
    
#     for d in ['train']:
# DatasetCatalog.register(d, lambda d: d = get_kitti_data_dict(dataset_root_dir, d))
# MetadataCatalog.get(d).thing_classes = ["person"]


# dataset_dicts = cuboid_annot2detectron_format(dataset_dir)
# # sample = random.sample(dataset_dicts, 3)
# i = 0



# for d in random.sample(DatasetCatalog.get('cuboid_dataset_train'), 5):
#     img = cv2.imread(d['file_name'])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=cuboid_metadata_train, scale=1)
#     out = visualizer.draw_dataset_dict(d)
#     # out = visualizer.draw_instance_predictions(dataset_dicts["annotations"].to("cpu"))
#     final_img = cv2.resize(out.get_image()[:, :, ::-1], (500,500))
#     cv2.imshow('2D BB', final_img)
#     # i=i+1
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    