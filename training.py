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

from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader

from my_loss_eval_hook import LossEvalHook
from detectron2.data import DatasetMapper

from detectron2.engine import HookBase
import detectron2.utils.comm as comm
from detectron2.data import build_detection_train_loader
import torch


#training
def do_training(train):
    cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("cuboid_dataset_train",)
    cfg.DATASETS.TEST = ("cuboid_dataset_val",) #used to obtain validation loss during training - do Not remove
    cfg.TEST.EVAL_PERIOD = 300 #number of iterations at which evaluation is run (to obtain validation loss) - It calls the evaluator, if specified
    cfg.DATALOADER.NUM_WORKERS = 8 #number of dataloading threads   
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 8 #needed since default number of keypoints is 17 in COCO dataset (for human pose estimation)
    cfg.TEST.KEYPOINT_OKS_SIGMAS = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.6]#same reason as for NUM_KEYPOINTS but for the evaluation part
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.0001 #0.00025 
    cfg.SOLVER.MAX_ITER = 30000  #300 iterations sufficient for mini dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256 #128 #number of ROIs to sample for training Fast RCNN head. sufficient for mini dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (cuboid)
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True #False -> images without annotation are Not removed during training
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # my_trainer = MyTrainer(cfg)
    
    # if train:
    #     my_trainer.resume_or_load(resume=False)
    #     my_trainer.train() 
    # else:
    #     my_trainer.resume_or_load(resume=True)
    # print(cfg.dump())
    # return my_trainer, cfg
    
    trainer = DefaultTrainer(cfg)
    val_loss = ValidationLoss(cfg)  
    trainer.register_hooks([val_loss])
    # swap the order of PeriodicWriter and ValidationLoss
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]
    if train:
        trainer.resume_or_load(resume=False)
        trainer.train() 
    else:
        trainer.resume_or_load(resume=True)
    print(cfg.dump())
    return trainer, cfg


# class MyTrainer(DefaultTrainer):
#     '''
#     subclass of DefaultTrainer class
#     '''
#     @classmethod
#     def build_evaluator(cls, cfg, dataset_name, output_folder=None):
#         if output_folder is None:
#             output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
#         return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
#     def build_hooks(self):
#         hooks = super().build_hooks()
#         hooks.insert(-1,LossEvalHook(
#             self.cfg.TEST.EVAL_PERIOD,
#             self.model,
#             build_detection_test_loader(
#                 self.cfg,
#                 self.cfg.DATASETS.TEST[0],
#                 DatasetMapper(self.cfg,True)
#             )
#         ))
#         return hooks 
    

class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
        self._loader = iter(build_detection_train_loader(self.cfg))
        
    def after_step(self):
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.trainer.model(data)
            
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {"val_" + k: v.item() for k, v in 
                                  comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced, 
                                                  **loss_dict_reduced)
    

if __name__=='__main__':
    do_training(train=True)    





