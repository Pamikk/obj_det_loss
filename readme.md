# Object detection loss functions and non-maximum suppressions

This repo will summarize and implement current loss functions and non-masimums uppression methods came up for object detection.

All methods will be evaluated on VOC2012 with same framework(currently YOLOv3).

Our goal is to analyze different tricks.

To do List

+ [ ] Mirgrate original dataset to VOC 2012
+ [ ] Revise codes to be more readable and concise
+ [ ] Loss_Funcs
  + [ ] bbox loss 
    + [ ] Anchor-based Loss
    + [ ] Regression Loss
    + [ ] IOU Loss
  + [ ] classification loss
+ [ ] Non-maximum-suppression
  + [ ] Hard NMS
  + [ ] Soft NMS
  + [ ]    