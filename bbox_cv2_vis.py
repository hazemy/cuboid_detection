#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2


def draw_bbox(img_dir, bbox):
    '''
    draws 2D bbox for 1 image
    '''
    img = cv2.imread(img_dir, cv2.IMREAD_COLOR)    
    start_pt = tuple([int(i) for i in bbox[0][0]])
    end_pt = tuple([int(i) for i in bbox[0][2]])
    print('Box coor: {}'.format(start_pt))
    # Blue color in BGR 
    color = (255, 0, 0) 
    # Line thickness of 2 px 
    thickness = 2
    img_pred = cv2.rectangle(img, start_pt, end_pt, color, thickness)
    cv2.imshow('GT BB', img_pred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()