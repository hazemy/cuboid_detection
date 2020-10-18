#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 01:38:41 2020

@author: porthos
"""

from detectron2.utils.logger import setup_logger
setup_logger(output='./output/log.txt') #save logs
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
# from detectron2.modeling import build_model
import os
# import yaml

from detectron2.evaluation import COCOEvaluator
# from detectron2.data import build_detection_test_loader

# from my_loss_eval_hook import LossEvalHook
# from detectron2.data import DatasetMapper

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
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_file(model_zoo.get_config_file("Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml"))
    cfg.DATASETS.TRAIN = ("cuboid_dataset_train",)
    cfg.DATASETS.TEST = ("cuboid_dataset_val",) #used by evaluator- do Not remove
    cfg.TEST.EVAL_PERIOD = 300 #number of iterations at which evaluation is run (Not relevant to validation loss calculation)
    cfg.SOLVER.CHECKPOINT_PERIOD = 3000
    cfg.DATALOADER.NUM_WORKERS = 2 #number of dataloading threads   
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml")
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.KEYPOINT_ON = True
    cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 0.5 #Keypoint AP degrades (though box AP improves) when using plain L1 loss (i.e: value = 0.0)
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1500 #1000 proposals per-image is found to hurt box AP
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 8 #needed since default number of keypoints is 17 in COCO dataset (for human pose estimation)
    cfg.TEST.KEYPOINT_OKS_SIGMAS = [0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.107, 0.107] #values show the importance of each keypoint location
                                        #smaller=more precise - coco smallest and largest sigmas for human keypoints are used
                                        #6th & 7th are assumed to be the usually hidden ones when having a vertical front face
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001 #0.0001 #0.00025 
    cfg.SOLVER.MAX_ITER = 5500  #300 iterations sufficient for mini dataset
    cfg.SOLVER.GAMMA = 0.1 #lr decay factor (in multistep LR scheduler)
    cfg.SOLVER.STEPS = [3500, 4500] #iteration milestones for reducing the lr (by gamma)
    # cfg.SOLVER.WARMUP_FACTOR = 0.0001 #start with a fraction of the learning rate for a number of iterations (warmup)
    # cfg.SOLVER.WARMUP_ITERS = 500 #warmup helps at initially avoiding learning irrelevant features
    # cfg.SOLVER.NESTEROV = 0.9 
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256 #128 #number of ROIs to sample for training Fast RCNN head. sufficient for mini dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (cuboid)
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True #False -> images without annotation are Not removed during training
    cfg.MODEL.PIXEL_MEAN = [124.396, 121.658, 110.113] #BGR
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    my_trainer = MyTrainer(cfg)
    
    # trainer = DefaultTrainer(cfg)
    val_loss = ValidationLoss(cfg) #runs every 20 iterations by default (rate related to tensorboard writing)
    my_trainer.register_hooks([val_loss])
    # swap the order of PeriodicWriter and ValidationLoss
    my_trainer._hooks = my_trainer._hooks[:-2] + my_trainer._hooks[-2:][::-1]
    if train:
        my_trainer.resume_or_load(resume=False)
        my_trainer.train() 
    else:
        my_trainer.resume_or_load(resume=True)
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        
    with open("./output/cfg_dump.txt", 'w') as file:
        file.write(cfg.dump())
    # print(cfg.dump())
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
    
    # def build_hooks(self):
    #     hooks = super().build_hooks()
    #     hooks.insert(-1,LossEvalHook(
    #         self.cfg.TEST.EVAL_PERIOD,
    #         self.model,
    #         build_detection_test_loader(
    #             self.cfg,
    #             self.cfg.DATASETS.TEST[0],
    #             DatasetMapper(self.cfg,True)
    #         )
    #     ))
    #     return hooks 
    
    @classmethod
    def build_train_loader(cls, cfg):
        dataloader_outputs = build_detection_train_loader(cfg)
        return dataloader_outputs
    

class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        # self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
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





