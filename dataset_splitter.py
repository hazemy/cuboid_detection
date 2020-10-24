#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:43:01 2020

@author: porthos
"""


from sklearn.model_selection import train_test_split
# from cuboid_annot2detectron_format import cuboid_annot2detectron_format
from format_converter import convert2detectron_format
from annot_processor import merge_annot_files


def split_dataset(dataset_list, train_ratio, val_ratio):
    '''
    splits dataset into train, val, and test datasets with each comprised of positive
    (those containing at 1 cuboid instance) and negative images. The ratio of positive images
    in each dataset is with respect to the number of positive images available in the
    dataset (and not the entire dataset size). Each dataset is complemented with negative 
    images so that its desired size (determined by train_ratio and val_ratio) can be reached.
    '''
        
    train_size = int(train_ratio * len(dataset_list))
    val_size = int(val_ratio * len(dataset_list))
    
    pos_dataset, neg_dataset = split_pos_neg(dataset_list)
    pos_train_ratio = train_ratio #ratio of +ve to -ve samples in train dataset
    pos_val_ratio = val_ratio
    pos_train_size = int(pos_train_ratio * len(pos_dataset))
    pos_val_size = int(pos_val_ratio * len(pos_dataset))
    
    pos_train_val_size = pos_train_size + pos_val_size 
    # shuffled_data = shuffle(dataset_list)
    train_val_size = train_size + val_size #worksround for splitting into 3 sets using sklearn fn
    
    #random_state field controls the seed for shuffling (same seed would give the same shuffled output (choose num between 0 and 42))
    # train_val_dataset, test_dataset = train_test_split(dataset_list, train_size=train_size, shuffle=True, random_state=42)
    pos_train_val_dataset, pos_test_dataset = train_test_split(pos_dataset, train_size=pos_train_val_size, shuffle=True, random_state=42)
    pos_train_dataset, pos_val_dataset = train_test_split(pos_train_val_dataset, train_size=pos_train_size, shuffle=True, random_state=42)
    
    if len(neg_dataset) > 0:
        neg_train_val_size = train_val_size - pos_train_val_size
        neg_train_size = train_size - pos_train_size
        neg_train_val_dataset, neg_test_dataset = train_test_split(neg_dataset, train_size=neg_train_val_size, shuffle=True, random_state=42)
        neg_train_dataset, neg_val_dataset = train_test_split(neg_train_val_dataset, train_size=neg_train_size, shuffle=True, random_state=42)
        
        ratio = 5/5
        cut_idx_train_pos = int((ratio)*len(pos_train_dataset))
        cut_idx_val_pos = int((ratio)*len(pos_val_dataset))
        cut_idx_test_pos = int((ratio)*len(pos_test_dataset))
        cut_idx_train_neg = int((ratio)*len(neg_train_dataset))
        cut_idx_val_neg = int((ratio)*len(neg_val_dataset))
        cut_idx_test_neg = int((ratio)*len(neg_test_dataset))
        
        train_dataset = pos_train_dataset[:cut_idx_train_pos] + neg_train_dataset[:cut_idx_train_neg]
        val_dataset = pos_val_dataset[:cut_idx_val_pos] + neg_val_dataset[:cut_idx_val_neg]
        test_dataset = pos_test_dataset[:cut_idx_test_pos] + neg_test_dataset[:cut_idx_test_neg]
    else:
        ratio = 5/5
        cut_idx_train_pos = int((ratio)*len(pos_train_dataset))
        cut_idx_val_pos = int((ratio)*len(pos_val_dataset))
        cut_idx_test_pos = int((ratio)*len(pos_test_dataset))
        
        train_dataset = pos_train_dataset[:cut_idx_train_pos]
        val_dataset = pos_val_dataset[:cut_idx_val_pos]
        test_dataset = pos_test_dataset[:cut_idx_test_pos]
   
    return (train_dataset, val_dataset, test_dataset)
    

def split_pos_neg(dataset_list):
    pos_dataset = []
    neg_dataset = []
    for entry in dataset_list:
        if entry['annotations']: #not empty
            pos_dataset.append(entry)
        else:
            neg_dataset.append(entry)
    return (pos_dataset, neg_dataset)


    
if __name__ == '__main__':
    annot_file_dir = '/home/porthos/masters_thesis/datasets/mini_dataset/mini_dataset_state.json'
    annot_files_dir_list = [annot_file_dir]
    annot_files_merged = merge_annot_files(annot_files_dir_list)
    images_dir = '/home/porthos/masters_thesis/datasets/mini_dataset/images'
    dataset_list = convert2detectron_format(annot_files_merged, images_dir)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset_list, 0.6, 0.2)    
    
    
    
