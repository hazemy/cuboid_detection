#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:43:01 2020

@author: porthos
"""

import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# from cuboid_annot2detectron_format import cuboid_annot2detectron_format
from json2detectron_format import json2detectron_format


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
    
    neg_train_val_size = train_val_size - pos_train_val_size
    neg_train_size = train_size - pos_train_size
    neg_train_val_dataset, neg_test_dataset = train_test_split(neg_dataset, train_size=neg_train_val_size, shuffle=True, random_state=42)
    neg_train_dataset, neg_val_dataset = train_test_split(neg_train_val_dataset, train_size=neg_train_size, shuffle=True, random_state=42)
    
    train_dataset = pos_train_dataset + neg_train_dataset
    val_dataset = pos_val_dataset + neg_val_dataset
    test_dataset = pos_test_dataset + neg_test_dataset
    # train_size = t_size #workaround for splitting into 3 sets using sklearn fn
    # train_dataset, val_dataset = train_test_split(train_val_dataset, train_size=train_size, shuffle=True, random_state=42)
    
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
    # dataset_list = '/home/porthos/masters_thesis/datasets/data_release/data_release/cuboid'
    annot_file = '/home/porthos/masters_thesis/datasets/dataset_all/state_partial.json'   
    # dataset_list = cuboid_annot2detectron_format(dataset_dir)
    dataset_list = json2detectron_format(annot_file)
    # pos_dataset, neg_dataset = split_pos_neg(dataset_list)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset_list, 0.6, 0.2)    
    
    
    
