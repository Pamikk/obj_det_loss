
#Train Setting
class Config:
    def __init__(self,mode='train'):
        #Path Setting
        self.test_img_path='../dataset/VOCdevkit/VOC2007/JPEGImages'
        self.img_path = '../dataset/VOCdevkit/VOC2007/JPEGImages'
        self.checkpoint='../checkpoints'
        self.cls_num = 20        
        self.res = 50
        self.sizes = [416]
        self.sizes_w = [1]
        self.nms_threshold = 0.5
        self.dc_threshold = 0.1
        #loss args
        #self.anchors = [[0.26533935,0.33382434],[0.66550966,0.56042827],[0.0880948,0.11774004]] #w,h normalized by max size
        #self.anchors = [[0.76822971,0.57259308],[0.39598597,0.47268035],[0.20632625,0.26720238],[0.07779112,0.10330848]] 
        self.anchors = [[0.053458141269278926, 0.07862022420023637],[0.1444091477545787, 0.11565781451235851],[0.1151994735538117, 0.24522582628542158],
           [0.31460831063222794, 0.2242885185476659],[0.21583068081593598, 0.4268351012487999],[0.38056995625914797, 0.5435495510304343],
           [0.6648272903930105, 0.3314712916726237],[0.5931101292049206, 0.7206548935846065],[0.8799995870003063, 0.5926446052236977]]
        self.anchor_divide=[(6,7,8),(3,4,5),(0,1,2)]
        self.anchor_num = len(self.anchors)
        self.obj_scale = 1
        self.noobj_scale = 0.5
        self.ignore_threshold = 0.5
        self.bs = 1        
        
        if mode=='train':
            self.file='./pre_data/train.json'
            self.bs = 32 # batch size
            #augmentation parameter
            self.rot = 0
            self.crop = 0.2
            self.valid_scale = 0.25
            #train_setting
            self.lr = 0.1
            self.weight_decay=5e-4
            self.min_lr = 1e-3
            self.lr_factor = 0.25
            #exp_setting
            self.save_every_k_epoch = 10

        elif mode=='val':
            self.file = './pre_data/val.json'
        elif mode=='trainval':
            self.file = './pre_data/trainval.json'
        elif mode=='test':
            self.file = './pre_data/trainval.json'
        
