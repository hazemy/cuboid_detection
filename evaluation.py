#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 01:38:41 2020

@author: porthos
"""


from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from training import do_training




def do_evaluation(dataset, trainer, cfg):
    cfg.TEST.KEYPOINT_OKS_SIGMAS = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.6] #TODO: tune evaluator
    evaluator = COCOEvaluator(dataset, cfg, False, output_dir="./output/")
    data_loader = build_detection_test_loader(cfg, dataset)
    print(inference_on_dataset(trainer.model, data_loader, evaluator)) #takes inputs (through the dataloader-2nd arg), 
                                       #outputs of the trained model(1st arg), and the method evaluation (evaluator-3rd arg)


if __name__=='__main__':
    trainer, cfg = do_training(train=False)
    dataset = "cuboid_dataset_test"
    do_evaluation(dataset, trainer, cfg)

