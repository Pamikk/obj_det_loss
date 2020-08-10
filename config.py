
#Train Setting
class Config:
    def __init__(self,mode='train'):
        #Path Setting
        self.test_img_path='../../dataset/global-wheat/test'
        self.img_path = '../../dataset/global-wheat/train'
        self.checkpoint='../../checkpoints'
        self.inp_size = (256,256)
        self.int_shape = 5
        self.grid = (16,16) #inp//4
        self.cls_num = 40 # Ax5 as mentioned in YOLO,ignore cls_score
        self.res = 50
        self.RGB_mean = [80.31413238,80.7378002,54.63867023]
        self.nms_threshold = 0.5
        self.dc_threshold = 0.01
        if mode=='train':
            self.file='../../dataset/global-wheat/train.json'
            self.bs = 24 # batch size
            #augmentation parameter
            self.rot = 10
            #self.scale = 0.25
            self.crop = 0.2
            self.flip = True
            self.valid_scale = 0.25
            self.sigmas =[(1,1),(3,3),(5,5),(9,9)]
            #train_setting
            self.lr = 0.1
            self.weight_decay=5e-4
            self.min_lr = 5e-6
            self.lr_factor = 0.1
            #exp_setting
            self.save_every_k_epoch = 10

        elif mode=='val':
            self.file = '../../dataset/global-wheat/val.json'
            self.bs = 1
        elif mode=='trainval':
            self.file = '../../dataset/global-wheat/train_subset.json'
            self.bs = 1
        
