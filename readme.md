# Object detection loss functions and non-maximum suppressions Survey

This repo will summarize and implement current loss functions and non-maximum suppression methods came up for object detection.

All methods will be evaluated on VOC2012
+ Ref:https://github.com/eriklindernoren/PyTorch-YOLOv3
  + Thanks eriklindernoren's share, it's really save me much time while debuging
  + models.py and parse_config and some other function are copied from his repo to help me check my reproduce correctness
  + you can use cmp_network parameter and add sth like this to Loss API to check outputs/structure of the network and the loss
+ A little tip:
  + watch out every hyperparameter while reporducing codes from scratch, even a little difference can make you spending half of a year to debug:( 
+ TBH, still working on hyperparameters tuning, but the whole code should work fine, someday.
## Template Code Structure for Deep Learning
  + Data
    + dataset annotation
    + dataset pre-processing/analyze
  + Models
    + utils - necessary function
      + maybe move evaluation metric here?
    + network
    + loss
    + backbone network
    + network configuration
  + dataProcessing
    + dataloader for train and test
  + Utils
    + logger
    + evaluation/metric
    + non-maximum-supression
  + visualization - jupyter notebook
  + Trainer - APIs
    + save/load weights
    + lr scheduler
    + optimizer(sometimes need to pass in network for things like GAN)    
  + train
  + test
  + evaluation - evaluate predicted result
  + config - configurate parpameters
## To do List
+ [x] Revise codes to be more readable and concise
+ [x] Loss_Funcs
  + [x] bbox loss
    + [x] Anchor-based Loss
      + [x] YOLOv3-based
        + [x] Regression Loss
        + [x] IOU Loss
        + [x] GIOU Loss$^{[1]}$#deal with gradient vanish caused by IOU is zero for non-overlap
        + [x] Combined regression with GIOU
  + [x] classification loss
     + [x]dice loss$^{[2]}$ #help deal with class imbalance
  + [ ] Anaylysis
  + [ ] Innovation
+ [x] Non-maximum-suppression
  + [x] Hard NMS
  + [x] Soft NMS$^{[3]}$

[1]:"Generalized Intersection over Union: A Metric and A Loss for BOunding Box Regression":https://giou.stanford.edu/GIoU.pdf
[2]:"v-net loss"
[3]:"Soft-NMS -- Improving Object Detection With One Line of Code":https://arxiv.org/pdf/1704.04503.pdf
