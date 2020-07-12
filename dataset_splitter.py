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
from cuboid_annot2detectron_format import cuboid_annot2detectron_format


def split_dataset(dataset_list, train_ratio, val_ratio):
    '''
    train and val ratios are with respect to dataset size
    '''
    t_size = int(train_ratio * len(dataset_list))
    v_size = int(val_ratio * len(dataset_list))
    # shuffled_data = shuffle(dataset_list)
    train_size = t_size + v_size #worksround for splitting into 3 sets using sklearn fn
    #random_state field controls the seed for shuffling (same seed would give the same shuffled output (choose num between 0 and 42))
    train_val_dataset, test_dataset = train_test_split(dataset_list, train_size=train_size, shuffle=True, random_state=42)
    train_size = t_size #worksround for splitting into 3 sets using sklearn fn
    train_dataset, val_dataset = train_test_split(train_val_dataset, train_size=train_size, shuffle=True, random_state=42)
    
    return (train_dataset, val_dataset, test_dataset)
    
    
    
if __name__ == '__main__':
    dataset_dir = '/home/porthos/masters_thesis/datasets/data_release/data_release/cuboid'
    dataset_list = cuboid_annot2detectron_format(dataset_dir)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset_list, 0.6, 0.2)
