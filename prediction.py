#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 01:38:41 2020

@author: porthos
"""


from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.checkpoint import DetectionCheckpointer
import cv2, os
from training import do_training


from detectron2.config import get_cfg


def get_predictor(dataset, cfg):  
    # cfg = get_cfg()
    # model_path = cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
    # model_path = cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
    model_path = cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    # cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0014999.pth")
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 #custom testing threshold for model
    cfg.DATASETS.TEST = dataset
    predictor = DefaultPredictor(cfg)
    return predictor


#prediction / inference (& visualization)
def do_prediction_and_visualization(dataset, cfg):
    dataset_dicts = DatasetCatalog.get(dataset)
    metadata=MetadataCatalog.get(dataset)
    # label=MetadataCatalog.get('cuboid_dataset_val').thing_classes 
    predictor = get_predictor(dataset, cfg)
    # for d in random.sample(dataset_dicts, 50): 
    for d in dataset_dicts:   
        if d['annotations']: #display +ve images only
            im = cv2.imread(d["file_name"])
            outputs = predictor(im)
            # print(outputs)
            v = Visualizer(im[:, :, ::-1],
                           metadata=metadata,
                           scale=1, 
            )
            # print(outputs['instances'].pred_classes)
            output_instances = outputs['instances'].to('cpu')
            pred = output_instances.pred_classes
            # print(pred.tolist())
            classes=[] 
            for i in range(len(pred.tolist())):
                classes.append('cuboid')
            scores = output_instances.scores
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(classes, scores)]
            # gt = v.draw_dataset_dict(d) #display ground truth annotation
            out = v.overlay_instances(boxes=output_instances.pred_boxes, labels=labels, keypoints=output_instances.pred_keypoints)
            final_img = cv2.resize(out.get_image()[:, :, ::-1], (900,900))
            cv2.imshow('Predication & GT:   ' + d['image_id'] + '.jpg', final_img)
            # cv2.imshow('Predication & GT:   ', final_img)
            k = cv2.waitKey(0)
            if k == 27: #esc key for stop
                cv2.destroyAllWindows()
                break
            cv2.destroyAllWindows()  
                # print(output_instances.get_centers())
        
   
if __name__=='__main__':
    _, cfg = do_training(train=False)
    dataset = "cuboid_dataset_test"
    # dataset = "mock_dataset"
    do_prediction_and_visualization(dataset, cfg)
    # do_prediction_and_visualization(dataset)






