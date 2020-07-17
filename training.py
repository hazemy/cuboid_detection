#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 01:38:41 2020

@author: porthos
"""

from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
import os

# from detectron2.engine import DefaultPredictor
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import DatasetCatalog, MetadataCatalog
# import random, cv2

# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader


#training
def do_training(train):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("cuboid_dataset_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2 #number of dataloading threads   
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # number of ROIs to sample for training Fast RCNN head. faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (cuboid)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    if train:
        trainer.resume_or_load(resume=False)
    else:
        trainer.resume_or_load(resume=True)
    trainer.train() #a 'trained' trainer is needed by the evaluator (not enough to load weights to cfg)
    return trainer, cfg

# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold for this model
# cfg.DATASETS.TEST = ("cuboid_dataset_val",)
# predictor = DefaultPredictor(cfg)


# #prediction & visualization
# dataset = 'cuboid_dataset_val'
# dataset_dicts = DatasetCatalog.get(dataset)
# metadata=MetadataCatalog.get(dataset)
# # label=MetadataCatalog.get('cuboid_dataset_val').thing_classes
# # for d in random.sample(dataset_dicts, 3):  
# for d in dataset_dicts:    
#     im = cv2.imread(d["file_name"])
#     outputs = predictor(im)
#     # print(outputs)
#     v = Visualizer(im[:, :, ::-1],
#                     metadata=metadata,
#                     scale=1, 
#                     # instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
#     )
#     print('Instances are: {}'.format(outputs['instances']))
#     # print(outputs['instances'].pred_classes)
#     # output_instances = outputs['instances'].pred_boxes.to('cpu')
#     output_instances = outputs['instances'].to('cpu')
#     pred = output_instances.pred_classes
#     # print(pred.tolist())
#     # print(list(pred.size())[0])
#     classes=[]
#     for i in range(len(pred.tolist())):
#         classes.append('cuboid')
#     scores = output_instances.scores
#     labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(classes, scores)]
#     out = v.overlay_instances(boxes=output_instances.pred_boxes, labels=labels)
#     final_img = cv2.resize(out.get_image()[:, :, ::-1], (500,500))
#     cv2.imshow('2D BB', final_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()  
#     # print(output_instances.get_centers())


# #inference & evaluation
# evaluator = COCOEvaluator(dataset, cfg, False, output_dir="./output/")
# val_loader = build_detection_test_loader(cfg, dataset)
# print(inference_on_dataset(trainer.model, val_loader, evaluator))
    


if __name__=='__main__':
    do_training(train=True)    





