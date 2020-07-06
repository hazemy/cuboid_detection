# -*- coding: utf-8 -*-

import json
import scipy.io as spio
import numpy as np



def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects   
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

# matdata = loadmat('Annotations.mat')
# print(matdata['data'][0].object.type)
# print(matdata['data'][842].object)
# print(type(matdata['data'][6].object[2])) #numpy array of scipy mat_struct objects
# print((matdata['data'][6].object).size)
# print(type(matdata['data'][0].object))
# print(isinstance(matdata['data'][842].object, np.ndarray))

def mat2cuboid_annot (annot_file_dir):
    '''
    converts input mat file to a list of dicts
    ouput: cuboid annotation (intemediate format)
    '''
    
    annot_file = annot_file_dir.split('/')[-1]
    matdata = loadmat(annot_file)
    cuboid_dataset_list = [] #a list of dicts (for entire dataset)
    for i in range(len(matdata['data'])):
        '''
        Parses the obtained mat data (dict) into an easily accessible (proper) dict
        '''
        im_dict = {} #per image dict
        im_dict['image'] = matdata['data'][i].im
        cuboid_instances = [] #all cuboids in image
        
        #check if the object stuct field is a single scipy mat_struct object or a... 
        #non-empty numpy array of annotations (foregound annotations)
        if (not isinstance(matdata['data'][i].object, np.ndarray)):
            instance = {} #annotation needed for each instance of a cuboid
            instance['type'] = matdata['data'][i].object.type
            instance['flipped'] = matdata['data'][i].object.flipped
            instance['hard'] = matdata['data'][i].object.hard
            instance['position'] = matdata['data'][i].object.position #a 2D list of...
                                        #vertices' coordinates (keypoints)...
                                        # other fileds such bbox and category id should be added
            cuboid_instances.append(instance)
            im_dict['object'] = cuboid_instances
            cuboid_dataset_list.append(im_dict)
            continue
        if ((matdata['data'][i].object).size > 0):
            for j in range((matdata['data'][i].object).size):
                instance = {} #annotation needed for each instance of a cuboid
                instance['type'] = matdata['data'][i].object[j].type
                instance['flipped'] = matdata['data'][i].object[j].flipped
                instance['hard'] = matdata['data'][i].object[j].hard
                instance['position'] = matdata['data'][i].object[j].position 
                cuboid_instances.append(instance)
            im_dict['object'] = cuboid_instances
            cuboid_dataset_list.append(im_dict)
            continue
        im_dict['object'] = cuboid_instances #handles background annotations 
        cuboid_dataset_list.append(im_dict)
    return cuboid_dataset_list
        

if __name__ == '__main__':
    annot_file_dir = '/home/porthos/masters_thesis/datasets/data_release/data_release/cuboid/Annotations.mat'
    cuboid_annot = mat2cuboid_annot(annot_file_dir)
















