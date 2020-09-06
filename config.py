
import numpy as np
import random
import json

from stats import kmeans
anchors =[[25.037, 31.196], [50.498, 79.588], [78.5, 86.013], [124.951, 126.297], [147.095, 168.983], [191.18, 202.286], [267.642, 220.965], [324.666, 305.925], [363.639, 347.229]]# [[26.646, 39.172], [57.83, 57.849], [71.924, 112.002], [107.664, 122.6], [157.196, 164.851], [190.146, 212.889], [296.238, 270.184], [330.639, 294.094], [438.711, 358.986]]#VOC07
#anchors = 
dataset = 'VOC2012'
path =f'data/annotation_{dataset}.json' #annotation path for anchor calculation
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
            if bw<0 or bh<0:
                print(name,bbox)
                exit()
            t = max(w,h)
            if sizes == None:
                scale = t
            else:
                scale = random.choice(sizes)
            allb.append((bw/t*scale,bh/t*scale))
    km = kmeans(allb,k=num,max_iters=1000)
    km.initialization()
    km.iter(0)
    km.print_cs()
    anchors = km.get_centers()
    km.cal_all_dist()  
    return anchors,km
#Train Setting
class Config:
    def __init__(self,mode='train'):
        #Path Setting
        self.img_path = f'../dataset/VOCdevkit/{dataset}/JPEGImages'
        self.checkpoint='../checkpoints'
        self.cls_num = 20        
        self.res = 50
        self.size = 416
        self.multiscale = 3
        self.sizes = list(range(self.size-32*self.multiscale,self.size+32*self.multiscale+1,32)) 
        self.nms_threshold = 0.5
        self.dc_threshold = 0.4
        
        
        #loss args
        #self.anchors = [[0.26533935,0.33382434],[0.66550966,0.56042827],[0.0880948,0.11774004]] #w,h normalized by max size
        #self.anchors = [[0.76822971,0.57259308],[0.39598597,0.47268035],[0.20632625,0.26720238],[0.07779112,0.10330848]]
        self.anchors= anchors  
        self.anchor_divide=[(6,7,8),(3,4,5),(0,1,2)]
        self.anchor_num = len(self.anchors)
        
        self.bs = 8       
        self.pre_trained_path = '../network_weights'
        if mode=='train':
            self.file=f'./data/train_{dataset}.json'
            self.bs = 32 # batch size
            self.flip = True
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
            #loss hyp
            self.obj_scale = 0.5
            self.noobj_scale = 1
            self.ignore_threshold = 0.7
            self.match_threshold = 0.2#regard as match above this threshold

        elif mode=='val':
            self.file = f'./data/val_{dataset}.json'
        elif mode=='trainval':
            self.file = f'./data/trainval_{dataset}.json'
        elif mode=='test':
            self.file = f'./data/trainval_{dataset}.json'
        
