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
        + [x] IOU Loss
        + [x] GIOU Loss$^{[1]}$#deal with gradient vanish caused by IOU is zero for non-overlap
        + [x] Combined regression with GIOU
  + [ ] classification loss
     + [ ]dice loss$^{[2]}$ #help deal with class imbalance
+ [ ] Non-maximum-suppression
  + [x] Hard NMS
  + [x] Soft NMS$^{[3]}$

[1]:"Generalized Intersection over Union: A Metric and A Loss for BOunding Box Regression":https://giou.stanford.edu/GIoU.pdf
[2]:"v-net loss"
[3]:"Soft-NMS -- Improving Object Detection With One Line of Code":https://arxiv.org/pdf/1704.04503.pdf
