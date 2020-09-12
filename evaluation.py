#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 01:38:41 2020

@author: porthos
"""


from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
import os
from training import do_training




#evaluation
def do_evaluation(dataset, trainer, cfg):
    # cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.TEST.KEYPOINT_OKS_SIGMAS = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.6] #TODO: tune evaluator
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (cuboid)
    # trainer=DefaultTrainer(cfg)
    # trainer.resume_or_load(resume=True)
    # output_dir = os.path.join(cfg.OUTPUT_DIR, 'evaluation')
    evaluator = COCOEvaluator(dataset, cfg, False, output_dir="./output/")
    # evaluator = COCOEvaluator(dataset, cfg, False, output_dir=output_dir)
    data_loader = build_detection_test_loader(cfg, dataset)
    print(inference_on_dataset(trainer.model, data_loader, evaluator)) #takes inputs (through the dataloader-2nd arg), 
                                       #outputs of the trained model(1st arg), and the method evaluation (evaluator-3rd arg)



if __name__=='__main__':
    trainer, cfg = do_training(train=False)
    dataset = "cuboid_dataset_test"
    do_evaluation(dataset, trainer, cfg)

