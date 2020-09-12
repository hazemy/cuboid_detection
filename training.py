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

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from my_loss_eval_hook import LossEvalHook
from detectron2.data import DatasetMapper


#training
def do_training(train):
    cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("cuboid_dataset_train",)
    cfg.DATASETS.TEST = ("cuboid_dataset_val",) #used to obtain validation loss during training - do Not remove
    cfg.TEST.EVAL_PERIOD = 100 #number of iterations at which evaluation is run (to obtain validation loss) - It calls the evaluator, if specified
    cfg.DATALOADER.NUM_WORKERS = 4 #number of dataloading threads   
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 8 #needed since default number of keypoints is 17 in COCO dataset (for human pose estimation)
    cfg.TEST.KEYPOINT_OKS_SIGMAS = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.6]#same reason as for NUM_KEYPOINTS but for the evaluation part
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.0001 #0.00025 
    cfg.SOLVER.MAX_ITER = 7000  #300 iterations sufficient for mini dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 #128 #number of ROIs to sample for training Fast RCNN head. sufficient for mini dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (cuboid)
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True #False -> images without annotation are Not removed during training
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # trainer = DefaultTrainer(cfg)  
    my_trainer = MyTrainer(cfg)
    
    if train:
        # trainer.resume_or_load(resume=False)
        # trainer.train() 
        my_trainer.resume_or_load(resume=False)
        my_trainer.train() 
    else:
        # trainer.resume_or_load(resume=True)
        my_trainer.resume_or_load(resume=True)
    return my_trainer, cfg


class MyTrainer(DefaultTrainer):
    '''
    subclass of DefaultTrainer class
    '''
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks    
    

if __name__=='__main__':
    do_training(train=True)    





