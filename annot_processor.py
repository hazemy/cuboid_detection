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
    Input: a merged list of directories of all annotation files that are in json format.
    Ouput: a list of dicts containing the entries of the merged annoation files
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
    
def get_unique(annot_files_merged):
    ids = []
    for entry in annot_files_merged:
        image_id = entry['fileName'].split('/')[-1] #annot_id = image id in annotation file
        image_id = image_id.split('\\')[-1]
        image_id = image_id.split('.')[-2] #use last section of image directory as image id   
        ids.append(image_id)
    unique_ids = []
    unique_index = []
    duplicates_ids = []
    duplicates_index = []
    for index, i in enumerate(ids):
        if i not in unique_ids:
            unique_ids.append(i)
            unique_index.append(index)
        else:
            print('Duplicate found')
            duplicates_ids.append(i)
            duplicates_index.append(index)
    unique = []
    for index in unique_index:
        unique.append(annot_files_merged[index])
    save_dir = '/home/porthos/Desktop'
    with open(os.path.join(save_dir, 'state_filtered.json'), 'w') as write_file:
        json.dump(unique, write_file, indent=4, sort_keys=True)     
    return (unique, duplicates_ids, duplicates_index)
    
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
            print('Image {} was Not found'.format(image_id))
            counter = counter + 1
            # print('Total of {} images were Not found'.format(counter))
            missing.append(image_id)
    return missing

def amend_dir(annot_file_filtered):
    '''
    Modify annotations' directory to fit annotation tool path
    '''
    new_dir = 'C:/Users/Hazem/Desktop/image-annotation-tool-master/image-annotation-tool-master'
    annot_file_amended = annot_file_filtered[:]
    for annot in annot_file_amended:
        annot_id = annot['fileName'].split('/')[-1] #annot_id = image id in annotation file
        # annot_id = annot_id.split('\\')[-1]
        full_new_dir = os.path.join(new_dir, annot_id)
        annot['fileName'] = full_new_dir
    save_dir = '/home/porthos/masters_thesis'
    with open(os.path.join(save_dir, 'state_all_final.json'), 'w') as write_file:
        json.dump(annot_file_amended, write_file, indent=4, sort_keys=True) 
    return annot_file_amended


if __name__ =='__main__':
    # annot_files_dir_list = []
    # annot_file_dir_1 = '/home/porthos/masters_thesis/datasets/full_dataset/state_hazem.json'
    # annot_file_dir_2 = '/home/porthos/masters_thesis/datasets/full_dataset/state_mojtaba.json'   
    # annot_file_dir_3 = '/home/porthos/masters_thesis/datasets/full_dataset/state_frederick.json'   
    # annot_file_dir_4 = '/home/porthos/masters_thesis/datasets/full_dataset/state_ammar.json'   
    annot_file_dir = '/home/porthos/masters_thesis/datasets/full_dataset/state_all_revised.json'
    # annot_files_dir_list = [annot_file_dir_1, annot_file_dir_2, annot_file_dir_3, annot_file_dir_4]
    annot_files_dir_list = [annot_file_dir]
    annot_files_merged = merge_annot_files(annot_files_dir_list)
    
    images_dir = '/home/porthos/masters_thesis/datasets/full_dataset/images'
    unique, duplicates_ids, duplicates_index = get_unique(annot_files_merged)
    missing = check_missing(unique, images_dir)
    # annot_file_amended = amend_dir(annot_file_filtered)

    
