#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 21:43:42 2020

@author: porthos
"""

import json, os, copy
import collections


def merge_annot_files(annot_files_list):
    '''
    Input: a merged list of directories of all annotations that are in json format.
    '''
    annot_files_merged = []
    # save_dir = '/home/porthos/masters_thesis/datasets/full_dataset'
    for annot_file in annot_files_list:
        with open(annot_file) as f:
            annot_list = json.load(f)
        for annot in annot_list:
            annot_files_merged.append(annot)
    # with open(os.path.join(save_dir, 'state_all.json'), 'w') as write_file:
    #     json.dump(annot_file_all, write_file, indent=4, sort_keys=True) 
    return annot_files_merged

def check_dublicates(annot_files_merged):
    ids = []
    for entry in annot_files_merged:
        image_id = entry['fileName'].split('/')[-1] #annot_id = image id in annotation file
        image_id = image_id.split('\\')[-1]
        image_id = image_id.split('.')[-2] #use last section of image directory as image id   
        ids.append(image_id)
    seen = []
    duplicates = []
    duplicates_idx = []
    for idx, i in enumerate(ids):
        if i not in seen:
            seen.append(i)
        else:
            print('Duplicate found')
            duplicates.append(i)
            duplicates_idx.append(idx)
    return (seen, duplicates)
    

def check_missing(annot_files_merged, images_dir):
    # img_path = os.path.join(images_dir, 'images') #construct path to images folder. * used to force accepting lists
    img_list = os.listdir(images_dir)
    counter = 0
    missing = []
    for img in img_list:
        image_id = img.split('.')[:-1][0] 
        found = False
        for annot in annot_files_merged:
            annot_id = annot['fileName'].split('/')[-1] #annot_id = image id in annotation file
            annot_id = annot_id.split('\\')[-1]
            annot_id = annot_id.split('.')[-2] #use last section of image directory as image id    
            if image_id == annot_id:
                # print('Image {} was found'.format(image_id))
                found = True
                break
        if not found:
            # print('Image {} was Not found'.format(image_id))
            counter = counter + 1
            # print('Total of {} images were Not found'.format(counter))
            missing.append(image_id)
    return missing
            


if __name__ =='__main__':
    annot_files_dir_list = []
    annot_file_dir_1 = '/home/porthos/masters_thesis/datasets/full_dataset/state_hazem.json'
    annot_file_dir_2 = '/home/porthos/masters_thesis/datasets/full_dataset/state_mojtaba.json'   
    annot_file_dir_3= '/home/porthos/masters_thesis/datasets/full_dataset/state_frederick.json'   
    annot_file_dir_4 = '/home/porthos/masters_thesis/datasets/full_dataset/state_ammar.json'   
    annot_files_dir_list = [annot_file_dir_1, annot_file_dir_2, annot_file_dir_3, annot_file_dir_4]
    annot_files_merged = merge_annot_files(annot_files_dir_list)
    
    # annot_file_all = '/home/porthos/masters_thesis/datasets/full_dataset/state_all.json'
    images_dir = '/home/porthos/masters_thesis/datasets/full_dataset/images'
    seen, duplicates = check_dublicates(annot_files_merged)
    
