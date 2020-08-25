
#Train Setting
class Config:
    def __init__(self,mode='train'):
        #Path Setting
        self.test_img_path='../dataset/VOCdevkit/VOC2007/JPEGImages'
        self.img_path = '../dataset/VOCdevkit/VOC2007/JPEGImages'
        self.checkpoint='../checkpoints'
        self.cls_num = 20        
        self.res = 50
        self.sizes = [256]
        self.sizes_w = [1]
        self.nms_threshold = 0.5
        self.dc_threshold = 0.1
        #loss args
        #self.anchors = [[0.26533935,0.33382434],[0.66550966,0.56042827],[0.0880948,0.11774004]] #w,h normalized by max size
        #self.anchors = [[0.76822971,0.57259308],[0.39598597,0.47268035],[0.20632625,0.26720238],[0.07779112,0.10330848]] 
        self.anchors = [[0.30184952,0.49619923],[0.18320179,0.23295927],[0.72675921,0.63843793],[0.58477846,0.30150818],[0.07225798,0.09619811]]
        self.anchor_num = len(self.anchors)
        self.obj_scale = 1
        self.noobj_scale = 0.5
        self.ignore_threshold = 0.5
        self.bs = 1        
        
        if mode=='train':
            self.file='./pre_data/train.json'
            self.bs = 20 # batch size
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
        
