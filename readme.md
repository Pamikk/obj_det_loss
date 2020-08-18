# Object detection loss functions and non-maximum suppressions

This repo will summarize and implement current loss functions and non-maximum suppression methods came up for object detection.

All methods will be evaluated on VOC2007 with same framework(currently YOLOv3).

Our goal is to analyze different tricks.

To do List

+ [ ] Mirgrate original model to VOC 2007
+ [ ] Revise codes to be more readable and concise
+ [ ] Loss_Funcs
  + [ ] bbox loss 
    + [x] Anchor-based Loss
      + [x] YOLOv3-based 
        + [x] Regression Loss
        + [x] IOU Loss,GIOU Loss$^{[1]}$
        + [x] Combined regression with GIOU
  + [ ] classification loss
+ [ ] Non-maximum-suppression
  + [ ] Hard NMS
  + [ ] Soft NMS

[1]:"Generalized Intersection over Union: A Metric and A Loss for BOunding Box Regression":https://giou.stanford.edu/GIoU.pdf
