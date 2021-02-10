
import numpy as np
import random
import json

from stats import kmeans
anchors = [[10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],  [156,198],  [373,326]]
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
                scale = sizes
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
        self.size = 608
        self.multiscale = 3
        self.sizes = list(range(self.size-32*self.multiscale,self.size+32*self.multiscale+1,32)) 
        self.nms_threshold = 0.5
        self.dc_threshold = 0.95

        self.anchors= anchors  
        self.anchor_divide=[(6,7,8),(3,4,5),(0,1,2)]
        self.anchor_num = len(self.anchors)
        
        self.bs = 8       
        self.pre_trained_path = '../network_weights'
        if mode=='train':
            self.file=f'./data/train_{dataset}.json'
            self.bs = 32 # batch size
            
            #augmentation parameter
            self.flip = True
            self.rot = 20
            self.crop = 0.2
            self.trans = 0.2
            self.scale = 0.1
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
            self.obj_scale = 2
            self.noobj_scale = 10
            self.cls_scale = 10
            self.ignore_threshold = 0.7
            self.match_threshold = 0.01#regard as match above this threshold

        elif mode=='val':
            self.file = f'./data/val_{dataset}.json'
        elif mode=='trainval':
            self.file = f'./data/trainval_{dataset}.json'
        elif mode=='test':
            self.file = f'./data/trainval_{dataset}.json'
        
