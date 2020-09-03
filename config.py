
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
        self.sizes =  [self.size] #+ list(range(self.size-32*3,self.size-32*3+1,32)) 
        self.sizes_w = [1]*len(self.sizes)
        self.sizes_w[0]+=3  
        self.nms_threshold = 0.3
        self.dc_threshold = 0.1
        
        #loss args
        #self.anchors = [[0.26533935,0.33382434],[0.66550966,0.56042827],[0.0880948,0.11774004]] #w,h normalized by max size
        #self.anchors = [[0.76822971,0.57259308],[0.39598597,0.47268035],[0.20632625,0.26720238],[0.07779112,0.10330848]] 
        self.anchors=[[28.112, 38.894], [51.368, 59.001], [83.605, 117.711], [122.2, 125.505], [123.953, 139.515], [219.663, 227.468], [280.433, 237.059], [391.719, 298.828], [395.413, 361.471]]
        self.anchor_divide=[(6,7,8),(3,4,5),(0,1,2)]
        self.anchor_num = len(self.anchors)
        self.obj_scale = 2.5
        self.noobj_scale = 0.5
        self.ignore_threshold = 0.7
        self.bs = 2       
        self.pre_trained_path = '../network_weights'
        if mode=='train':
            self.file='./pre_data/train.json'
            self.bs = 32 # batch size
            self.flip = False
            #augmentation parameter
            self.rot = 0
            self.crop = 0.2
            self.valid_scale = 0.25
            #train_setting
            self.lr = 1e-3
            self.weight_decay=5e-4
            self.momentum = 0.9
            #lr_scheduler
            self.min_lr = 1e-7
            self.lr_factor = 0.25
            self.patience = 10
            #exp_setting
            self.save_every_k_epoch = 10
            self.val_every_k_epoch = 10
            self.adjust_lr = False

        elif mode=='val':
            self.file = './pre_data/val.json'
        elif mode=='trainval':
            self.file = './pre_data/trainval.json'
        elif mode=='test':
            self.file = './pre_data/trainval.json'
        
