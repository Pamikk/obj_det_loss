
#Train Setting
class Config:
    def __init__(self,mode='train'):
        #Path Setting
        self.test_img_path='../dataset/VOCdevkit/VOC2007/JPEGImages'
        self.img_path = '../dataset/VOCdevkit/VOC2007/JPEGImages'
        self.checkpoint='../checkpoints'
        self.cls_num = 20        
        self.res = 50
        self.sizes = [384,416,448]
        self.sizes_w = [1,1,1]
        self.nms_threshold = 0.5
        self.dc_threshold = 0.1
        #loss args
        #self.anchors = [[0.26533935,0.33382434],[0.66550966,0.56042827],[0.0880948,0.11774004]] #w,h normalized by max size
        #self.anchors = [[0.76822971,0.57259308],[0.39598597,0.47268035],[0.20632625,0.26720238],[0.07779112,0.10330848]] 
        self.anchors = [[26.619310998735777, 39.15455120101138],[71.77704517704518, 57.77313797313797],[57.81336725254394, 122.58395004625346],
           [156.65975718092983, 111.81196328101865],[107.69161406672679, 212.91523895401264],[190.1323076923077, 270.1545299145299],
           [329.9393296563428, 164.39669070852779],[296.2381738173817, 358.98624862486247],[438.50071873502634, 293.79731672256827]]
        self.anchor_divide=[(6,7,8),(3,4,5),(0,1,2)]
        self.anchor_num = len(self.anchors)
        self.obj_scale = 1
        self.noobj_scale = 0.5
        self.ignore_threshold = 0.5
        self.bs = 1        
        self.pre_trained_path = '../network_weights'
        if mode=='train':
            self.file='./pre_data/train.json'
            self.bs = 32 # batch size
            #augmentation parameter
            self.rot = 0
            self.crop = 0.2
            self.valid_scale = 0.25
            #train_setting
            self.lr = 0.001
            self.weight_decay=5e-4
            self.momentum = 0.9
            self.min_lr = 5e-6
            self.lr_factor = 0.25
            #exp_setting
            self.save_every_k_epoch = 10

        elif mode=='val':
            self.file = './pre_data/val.json'
        elif mode=='trainval':
            self.file = './pre_data/trainval.json'
        elif mode=='test':
            self.file = './pre_data/trainval.json'
        
