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

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader

# from MyTrainer import MyTrainer

#training
def do_training(train):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("cuboid_dataset_train",)
    cfg.DATASETS.TEST = ("cuboid_dataset_val",) #used to obtain validation loss during training - do Not remove
    cfg.TEST.EVAL_PERIOD = 100 #number of iterations at which evaluation is run (to obtain validation loss) - It calls the evaluator, if specified
    cfg.DATALOADER.NUM_WORKERS = 2 #number of dataloading threads   
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 3000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 #128   # number of ROIs to sample for training Fast RCNN head. faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (cuboid)
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False #False -> images without annotation are Not removed during training
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # trainer = DefaultTrainer(cfg)  
    my_trainer = MyTrainer(cfg)
    # evaluator = COCOEvaluator("cuboid_dataset_val", cfg, True, output_dir="./output/")
    # trainer.test(cfg=cfg, model=trainer.model, evaluators=[evaluator])

    if train:
        # trainer.resume_or_load(resume=False)
        # trainer.train() 
        my_trainer.resume_or_load(resume=False)
        my_trainer.train() 
    else:
        # trainer.resume_or_load(resume=True)
        my_trainer.resume_or_load(resume=True)
    # data_loader = build_detection_test_loader(cfg, "cuboid_dataset_val")
    # print(inference_on_dataset(trainer.model, data_loader, evaluator))
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





