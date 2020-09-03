
import numpy as np
import random
import json

from stats import kmeans
anchors = [[26.646, 39.172], [57.83, 57.849], [71.924, 112.002], [107.664, 122.6], [157.196, 164.851], [190.146, 212.889], [296.238, 270.184], [330.639, 294.094], [438.711, 358.986]]
path ='data/annotation_voc07.json' #annotation path for anchor calculation
def cal_anchors(sizes=None,num=9):
    #As in https://github.com/eriklindernoren/PyTorch-YOLOv3
    # randomly scale as sizes if sizes is not None    
    annos = json.load(open(path,'r'))
    allb = []
    for name in annos:
        anno = annos[name]
        size = anno['size']
        w,h,_ = size
        for bbox in anno['labels']:
            xmin,ymin,xmax,ymax = bbox[1:5]
            bw,bh = xmax-xmin,ymax-ymin
            t = max(w,h)
            if sizes == None:
                scale = t
            else:
                scale = random.choice(sizes)
            allb.append((bw/t*scale,bh/t*scale))
    km = kmeans(allb,k=num,max_iters=500)
    km.initialization()
    km.iter(0)
    km.print_cs()  
    return km.get_centers()
#Train Setting
class Config:
    def __init__(self,mode='train'):
        #Path Setting
        self.test_img_path='../dataset/VOCdevkit/VOC2007/JPEGImages'
        self.img_path = '../dataset/VOCdevkit/VOC2007/JPEGImages'
        self.checkpoint='../checkpoints'
        self.cls_num = 20        
        self.res = 50
        self.size = 416
        self.multiscale = 1
        self.sizes = list(range(self.size-32*self.multiscale,self.size-32*self.multiscale+1,32)) 
        self.nms_threshold = 0.5
        self.dc_threshold = 0.4
        
        #loss args
        #self.anchors = [[0.26533935,0.33382434],[0.66550966,0.56042827],[0.0880948,0.11774004]] #w,h normalized by max size
        #self.anchors = [[0.76822971,0.57259308],[0.39598597,0.47268035],[0.20632625,0.26720238],[0.07779112,0.10330848]]
        self.keep_anchors = True
        if self.keep_anchors:
            self.anchors= anchors  
        else:
            self.anchors =  cal_anchors(self.sizes)
        self.anchor_divide=[(6,7,8),(3,4,5),(0,1,2)]
        self.anchor_num = len(self.anchors)
        self.obj_scale = 1.5
        self.noobj_scale = 0.5
        self.ignore_threshold = 0.7
        self.bs = 8       
        self.pre_trained_path = '../network_weights'
        if mode=='train':
            self.file='./data/train.json'
            self.bs = 32 # batch size
            self.flip = False
            #augmentation parameter
            self.rot = 0
            self.crop = 0.2
            self.valid_scale = 0.25
            #train_setting
            self.lr = 0.01
            self.weight_decay=5e-4
            self.momentum = 0.9
            #lr_scheduler
            self.min_lr = 5e-5
            self.lr_factor = 0.25
            self.patience = 12
            #exp_setting
            self.save_every_k_epoch = 15
            self.val_every_k_epoch = 10
            self.adjust_lr = False

        elif mode=='val':
            self.file = './data/val.json'
        elif mode=='trainval':
            self.file = './data/trainval.json'
        elif mode=='test':
            self.file = './data/trainval.json'
        
