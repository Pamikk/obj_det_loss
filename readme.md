# Object detection loss functions and non-maximum suppressions

This repo will summarize and implement current loss functions and non-maximum suppression methods came up for object detection.

All methods will be evaluated on VOC2007 with the same framework(currently YOLOv3).

Our goal is to analyze different tricks.

Currently under training procedure, periodical conclusion will be soon updated.

+ this repo will not be updated in a short time( at least till the end of October).
+ In my observation: there are more classification errors than location errors due to class imbalance causing low AP. Now it shows that dice loss can not solve this well as it works in segmentation task.
+ As a result, I will work on classification part and debug about the pretained-weight loading part( it shows weight-load now does not work as expectation).
+ I have heard that YOLOv4 and YOLOv5 work better on wheat head detection than its performance on object detection. I guess it might be caused by the problem mentioned above(i.e. performance of YOLO are not limited by the bbox accuracy but the classification accuracy). I will study more on few-shot learning and debugging my current work to find true reason causing low AP(might just because I suck at hyper-parameter tuning)
+ Actually I get a not-bad result on the other repo(wheat-det,like about 0.63 on my own validation for YOLOv3-spp,giou loss)
+ I modify the classification activation function to softmax from sigmoid w/o practice.
+ All models are trained on colab.
+ Attention!!!  Current useful result:G-IOU loss worked really well although its gradient is hard to calculate by hands and don't use this repo.

To do List

+ [x] Mirgrate original model to VOC 2007
+ [x] Revise codes to be more readable and concise
+ [x] Loss_Funcs
  + [ ] bbox loss
    + [x] Anchor-based Loss
      + [x] YOLOv3-based
        + [x] Regression Loss
        + [x] IOU Loss
        + [x] GIOU Loss$^{[1]}$#deal with gradient vanish caused by IOU is zero for non-overlap
        + [x] Combined regression with GIOU
  + [x] classification loss
     + [x]dice loss$^{[2]}$ #help deal with class imbalance
+ [x] Non-maximum-suppression
  + [x] Hard NMS
  + [x] Soft NMS$^{[3]}$
+ [x] Add pretrain on VOC

[1]:"Generalized Intersection over Union: A Metric and A Loss for BOunding Box Regression":https://giou.stanford.edu/GIoU.pdf
[2]:"v-net loss"
[3]:"Soft-NMS -- Improving Object Detection With One Line of Code":https://arxiv.org/pdf/1704.04503.pdf
