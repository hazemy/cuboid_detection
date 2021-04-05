#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from detectron2.data import DatasetCatalog, MetadataCatalog
from format_converter import convert2detectron_format
from dataset_splitter import split_dataset
from annot_processor import merge_annot_files, get_unique, remove_faulty, check_visibilty



# annot_file_dir = '/home/porthos/masters_thesis/datasets/mini_dataset/mini_dataset_state.json'
# images_dir = '/home/porthos/masters_thesis/datasets/mini_dataset/images'
annot_file_dir_1 = '/home/porthos/masters_thesis/datasets/final_dataset/annotations_hazem_mod.json'
annot_file_dir_2 = '/home/porthos/masters_thesis/datasets/final_dataset/annotations_ammar.json'
annot_file_dir_3 = '/home/porthos/masters_thesis/datasets/final_dataset/annotations_pablo.json'
annot_file_dir_4 = '/home/porthos/masters_thesis/datasets/final_dataset/annotations_anas.json'
annot_file_dir_5 = '/home/porthos/masters_thesis/datasets/final_dataset/annotations_leonie.json'

images_dir = '/home/porthos/masters_thesis/datasets/final_dataset/images'
annot_files_dir_list = [annot_file_dir_1, annot_file_dir_2, annot_file_dir_3, annot_file_dir_4, annot_file_dir_5]
# annot_files_dir_list = [annot_file_dir_1, annot_file_dir_2, annot_file_dir_3, annot_file_dir_4]
annot_files_merged = merge_annot_files(annot_files_dir_list)

annot_file_filtered, _, _ = get_unique(annot_files_merged)
annot_file_corrected = remove_faulty(annot_file_filtered)
# check_visibilty(annot_file_filtered)
dataset_list = convert2detectron_format(annot_file_filtered, images_dir)
train_dataset, val_dataset, test_dataset = split_dataset(dataset_list, 0.6, 0.2)

DatasetCatalog.register('cuboid_dataset' + '_train', lambda: train_dataset)
MetadataCatalog.get('cuboid_dataset' + '_train').set(thing_classes=['cuboid'])  
MetadataCatalog.get('cuboid_dataset' + '_train').set(keypoint_names=['FUL', 'FUR', 'FLR', 'FLL', 'BUL', 'BUR', 'BLR', 'BLL'])      
# MetadataCatalog.get('cuboid_dataset' + '_train').set(keypoint_flip_map=[('FUL', 'BUR'), ('FLL', 'BLR'), ('BUL', 'BUL'), ('BLL', 'BLL'), ('FLR', 'FLR'), ('FUR', 'FUR')])    
# MetadataCatalog.get('cuboid_dataset' + '_train').set(keypoint_flip_map=[('FUL', 'FUL'), ('FLL', 'FLL'), ('BUL', 'BUL'), ('BUR', 'BUR'), ('BLL', 'BLL'), ('BLR', 'BLR'), ('FLR', 'FLR'), ('FUR', 'FUR')])    
MetadataCatalog.get('cuboid_dataset' + '_train').set(keypoint_flip_map=[('FUL', 'FUR'), ('FLL', 'FLR'), ('BUL', 'BUR'), ('BLL', 'BLR')])    
# MetadataCatalog.get('cuboid_dataset' + '_train').set(keypoint_flip_map=[])    
# MetadataCatalog.get('cuboid_dataset' + '_train').set(keypoint_connection_rules=[('FUL', 'FUR', (255,255,0)), ('FUR', 'FLR', (255,255,0)), ('FLR', 'FLL', (255,255,0)), ('FUL', 'FLL', (255,255,0)), ('BUL', 'BUR',(0,0,255)), ('BUR', 'BLR',(0,0,255)), \
                                                                                # ('BLR', 'BLL', (0,0,255)), ('BLL', 'BUL', (0,0,255)), ('FUL', 'BUL', (0,0,255)), ('FUR', 'BUR', (0,0,255)), ('FLR', 'BLR', (0,0,255)), ('FLL', 'BLL', (0,0,255))])
MetadataCatalog.get('cuboid_dataset' + '_train').set(keypoint_connection_rules=[('FUL', 'FUR', (255,127,14)), ('FUR', 'FLR', (255,127,14)), ('FLR', 'FLL', (255,127,14)), ('FUL', 'FLL', (255,127,14)), ('BUL', 'BUR',(31,119,180)), ('BUR', 'BLR',(31,119,180)), \
                                                                                ('BLR', 'BLL', (31,119,180)), ('BLL', 'BUL', (31,119,180)), ('FUL', 'BUL', (31,119,180)), ('FUR', 'BUR', (31,119,180)), ('FLR', 'BLR', (31,119,180)), ('FLL', 'BLL', (31,119,180))])
cuboid_metadata_train = MetadataCatalog.get("cuboid_dataset_train")
#TODO: Check id cuboid annotation exists

DatasetCatalog.register('cuboid_dataset' + '_val', lambda: val_dataset)
MetadataCatalog.get('cuboid_dataset' + '_val').set(thing_classes=['cuboid'])
MetadataCatalog.get('cuboid_dataset' + '_val').set(keypoint_names=['FUL', 'FUR', 'FLR', 'FLL', 'BUL', 'BUR', 'BLR', 'BLL'])      
# MetadataCatalog.get('cuboid_dataset' + '_val').set(keypoint_flip_map=[('FUL', 'BUR'), ('FLL', 'BLR'), ('BUL', 'BUL'), ('BLL', 'BLL'), ('FLR', 'FLR'), ('FUR', 'FUR')])      
# MetadataCatalog.get('cuboid_dataset' + '_val').set(keypoint_flip_map=[('FUL', 'FUL'), ('FLL', 'FLL'), ('BUL', 'BUL'), ('BUR', 'BUR'), ('BLL', 'BLL'), ('BLR', 'BLR'), ('FLR', 'FLR'), ('FUR', 'FUR')])    
MetadataCatalog.get('cuboid_dataset' + '_val').set(keypoint_flip_map=[('FUL', 'FUR'), ('FLL', 'FLR'), ('BUL', 'BUR'), ('BLL', 'BLR')])    
# MetadataCatalog.get('cuboid_dataset' + '_val').set(keypoint_flip_map=[])    
# MetadataCatalog.get('cuboid_dataset' + '_val').set(keypoint_connection_rules=[('FUL', 'FUR', (255,255,0)), ('FUR', 'FLR', (255,255,0)), ('FLR', 'FLL', (255,255,0)), ('FUL', 'FLL', (255,255,0)), ('BUL', 'BUR',(0,0,255)), ('BUR', 'BLR',(0,0,255)), \
                                                                                # ('BLR', 'BLL', (0,0,255)), ('BLL', 'BUL', (0,0,255)), ('FUL', 'BUL', (0,0,255)), ('FUR', 'BUR', (0,0,255)), ('FLR', 'BLR', (0,0,255)), ('FLL', 'BLL', (0,0,255))])
MetadataCatalog.get('cuboid_dataset' + '_val').set(keypoint_connection_rules=[('FUL', 'FUR', (255,127,14)), ('FUR', 'FLR', (255,127,14)), ('FLR', 'FLL', (255,127,14)), ('FUL', 'FLL', (255,127,14)), ('BUL', 'BUR',(31,119,180)), ('BUR', 'BLR',(31,119,180)), \
                                                                                ('BLR', 'BLL', (31,119,180)), ('BLL', 'BUL', (31,119,180)), ('FUL', 'BUL', (31,119,180)), ('FUR', 'BUR', (31,119,180)), ('FLR', 'BLR', (31,119,180)), ('FLL', 'BLL', (31,119,180))])
cuboid_metadata_val = MetadataCatalog.get("cuboid_dataset_val")
  
DatasetCatalog.register('cuboid_dataset' + '_test', lambda: test_dataset)
MetadataCatalog.get('cuboid_dataset' + '_test').set(thing_classes=['cuboid']) 
MetadataCatalog.get('cuboid_dataset' + '_test').set(keypoint_names=['FUL', 'FUR', 'FLR', 'FLL', 'BUL', 'BUR', 'BLR', 'BLL'])      
# MetadataCatalog.get('cuboid_dataset' + '_test').set(keypoint_flip_map=[('FUL', 'BUR'), ('FLL', 'BLR'), ('BUL', 'BUL'), ('BLL', 'BLL'), ('FLR', 'FLR'), ('FUR', 'FUR')])       
# MetadataCatalog.get('cuboid_dataset' + '_test').set(keypoint_flip_map=[('FUL', 'FUL'), ('FLL', 'FLL'), ('BUL', 'BUL'), ('BUR', 'BUR'), ('BLL', 'BLL'), ('BLR', 'BLR'), ('FLR', 'FLR'), ('FUR', 'FUR')])    
MetadataCatalog.get('cuboid_dataset' + '_test').set(keypoint_flip_map=[('FUL', 'FUR'), ('FLL', 'FLR'), ('BUL', 'BUR'), ('BLL', 'BLR')])    
# MetadataCatalog.get('cuboid_dataset' + '_test').set(keypoint_flip_map=[])    
# MetadataCatalog.get('cuboid_dataset' + '_test').set(keypoint_connection_rules=[('FUL', 'FUR', (255,255,0)), ('FUR', 'FLR', (255,255,0)), ('FLR', 'FLL', (255,255,0)), ('FUL', 'FLL', (255,255,0)), ('BUL', 'BUR',(0,0,255)), ('BUR', 'BLR',(0,0,255)), \
                                                                                # ('BLR', 'BLL', (0,0,255)), ('BLL', 'BUL', (0,0,255)), ('FUL', 'BUL', (0,0,255)), ('FUR', 'BUR', (0,0,255)), ('FLR', 'BLR', (0,0,255)), ('FLL', 'BLL', (0,0,255))])
MetadataCatalog.get('cuboid_dataset' + '_test').set(keypoint_connection_rules=[('FUL', 'FUR', (255,127,14)), ('FUR', 'FLR', (255,127,14)), ('FLR', 'FLL', (255,127,14)), ('FUL', 'FLL', (255,127,14)), ('BUL', 'BUR',(31,119,180)), ('BUR', 'BLR',(31,119,180)), \
                                                                                ('BLR', 'BLL', (31,119,180)), ('BLL', 'BUL', (31,119,180)), ('FUL', 'BUL', (31,119,180)), ('FUR', 'BUR', (31,119,180)), ('FLR', 'BLR', (31,119,180)), ('FLL', 'BLL', (31,119,180))])
cuboid_metadata_test = MetadataCatalog.get("cuboid_dataset_test") 
    
    
# #Mock Dataset COCO
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("mock_dataset", {}, "/home/porthos/Downloads/mock_dataset/image_info_test2017.json", "/home/porthos/Downloads/mock_dataset/test2017")
    
    