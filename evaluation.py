#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from training import do_training




def do_evaluation(dataset, trainer, cfg):
    sigmas = [0.025, 0.025, 0.025, 0.025, 0.025, 0.025, 0.107, 0.107] #TODO: tune evaluator
    # evaluator = COCOEvaluator(dataset, cfg, False, output_dir="./output/")
    evaluator = COCOEvaluator(dataset_name=dataset, tasks=("bbox","keypoints"), distributed=True, output_dir="./output/", use_fast_impl=True, kpt_oks_sigmas=sigmas)
    data_loader = build_detection_test_loader(cfg, dataset)
    print(inference_on_dataset(trainer.model, data_loader, evaluator)) #takes inputs (through the dataloader-2nd arg), 
                                       #outputs of the trained model(1st arg), and the method evaluation (evaluator-3rd arg)


if __name__=='__main__':
    trainer, cfg = do_training(train=False)
    dataset = "cuboid_dataset_val"
    do_evaluation(dataset, trainer, cfg)

