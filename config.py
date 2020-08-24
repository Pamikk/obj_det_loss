
#Train Setting
class Config:
    def __init__(self,mode='train'):
        #Path Setting
        self.test_img_path='../dataset/VOCdevkit/VOC2007/JPEGImages'
        self.img_path = '../dataset/VOCdevkit/VOC2007/JPEGImages'
        self.checkpoint='../checkpoints'
        self.cls_num = 20        
        self.res = 50
        self.sizes = range(256,513,32)
        self.nms_threshold = 0.5
        self.dc_threshold = 0.4
        #loss args
        self.anchors = [[0.18414403,0.30230376],[0.57590277,0.38793478],[0.7645372,0.79610235],[0.30366338,0.63555215],[0.07356203,0.12814362]] #(w,h),normalized
        self.anchor_num = len(self.anchors)
        self.obj_scale = 5
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
            self.lr = 0.01
            self.weight_decay=5e-4
            self.min_lr = 5e-5
            self.lr_factor = 0.1
            #exp_setting
            self.save_every_k_epoch = 10

        elif mode=='val':
            self.file = './pre_data/val.json'
        elif mode=='trainval':
            self.file = './pre_data/trainval.json'
        elif mode=='test':
            self.file = './pre_data/test.json'
        
