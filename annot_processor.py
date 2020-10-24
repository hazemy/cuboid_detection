#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 21:43:42 2020

@author: porthos
"""

import json, os


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
            image_id = annot['fileName'].split('/')[-1] #annot_id = image id in annotation file
            image_id = image_id.split('\\')[-1]
            image_id = image_id.split('.')[-2] #use last section of image directory as image id
            if 'Logistic' not in image_id: #removes logistic dataset temporarily for testing
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
    # save_dir = '/home/porthos/Desktop'
    # with open(os.path.join(save_dir, 'state_filtered.json'), 'w') as write_file:
    #     json.dump(unique, write_file, indent=0, sort_keys=True)     
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
    Modify annotations' directory to fit annotation tool path and replaces png extension with jpg
    in file name
    '''
    new_dir = 'C:/Users/Hazem/Desktop/image-annotation-tool-master/image-annotation-tool-master/images\\'
    annot_file_amended = annot_file_filtered[:]
    for annot in annot_file_amended:
        annot_id = annot['fileName'].split('/')[-1] #annot_id = image id in annotation file
        annot_id = annot_id.split('\\')[-1]
        annot_id = annot_id.split('.')[-2]
        # print(annot_id)
        # full_new_dir = os.path.join(new_dir, annot_id)
        full_new_dir = new_dir + annot_id + '.jpg'
        annot['fileName'] = full_new_dir
    save_dir = '/home/porthos/Desktop'
    # with open(os.path.join(save_dir, 'state_all_final.json'), 'w') as write_file:
    #     json.dump(annot_file_amended, fp=write_file, indent=5) 
    return annot_file_amended

def remove_faulty(annot_file_filtered):
    '''
    Removes faulty annotations (images with extra annotations)
    '''
    #TODO: double check that these images are actually faulty
    #order: me (0-3) - anas (4-10) - ammar (11-14) - pablo (15-end & Logistic149 is missing cubes)
    faulty_list = [
                   'c_cellblock_cellblock_000203',\
                   'f_firebreak_firebreak_000015',\
                   's_semidesert_semidesert_000066',\
                   'f_floating_dock_floating_dock_000050',\
                   'f_foothill_foothill_000006',\
                   'h_hothouse_indoor_hothouse_000006',\
                   'i_ice_field_ice_field_000013',\
                   'i_ice_hockey_rink_indoor_ice-hockey_rink_000013',\
                   'o_opera_outdoor_opera_house_000720',\
                   't_television_room_television_room_000006',\
                   'b_bay_bay_000004',\
                   'l_ledge_ledge_000049',\
                   'c_country_road_roadway_000018',\
                   't_t-bar_lift_t-bar_lift_000018',
                   's_seaside_seaside_000779',\
                   't_t-bar_lift_t-bar_lift_000132',\
                   's_shore_shore_000016',\
                   'w_wetland_wetland_000016',\
                   'e_estuary_estuary_000010',\
                   'm_millpond_millpond_000009',\
                   's_seaside_seaside_000009',\
                   'a_apple_orchard_apple_orchard_000004'
                  ]
    for annot in annot_file_filtered:
        annot_id = annot['fileName'].split('/')[-1] #annot_id = image id in annotation file
        annot_id = annot_id.split('\\')[-1]
        annot_id = annot_id.split('.')[-2]
        if annot_id in faulty_list:
            # print('Faulty Image Found: {}'.format(annot_id))
            annot['squares'] = []
            annot['cubes'] = []
    return annot_file_filtered
    
def check_visibilty(annot_file):
    vis_list = [0, 1, 2]
    for annot in annot_file:
        annot_id = annot['fileName'].split('/')[-1] #annot_id = image id in annotation file
        annot_id = annot_id.split('\\')[-1]
        annot_id = annot_id.split('.')[-2]
        cubes = annot['cubes']
        for cube in cubes:
            for corner in cube:
                assert (corner[2] in vis_list), 'Invalid Visibility Value in {} at corner {}'.format(annot_id, cube.index(corner))
                    


if __name__ =='__main__':
    annot_file_dir_1 = '/home/porthos/masters_thesis/datasets/final_dataset/annotations_hazem_mod.json'
    annot_file_dir_2 = '/home/porthos/masters_thesis/datasets/final_dataset/annotations_ammar.json'
    annot_file_dir_3 = '/home/porthos/masters_thesis/datasets/final_dataset/annotations_pablo.json'
    annot_file_dir_4 = '/home/porthos/masters_thesis/datasets/final_dataset/annotations_anas.json'
    annot_file_dir_5 = '/home/porthos/masters_thesis/datasets/final_dataset/annotations_leonie.json'
    
    images_dir = '/home/porthos/masters_thesis/datasets/final_dataset/images'
    annot_files_dir_list = [annot_file_dir_1, annot_file_dir_2, annot_file_dir_3, annot_file_dir_4, annot_file_dir_5]
    annot_files_merged = merge_annot_files(annot_files_dir_list)
    
    unique, duplicates_ids, duplicates_index = get_unique(annot_files_merged)
    missing = check_missing(unique, images_dir)
    annot_file_amended = amend_dir(unique)
    annot_file_corrected = remove_faulty(annot_file_amended)
    check_visibilty(annot_file_corrected)
    

    
