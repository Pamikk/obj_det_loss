
#Train Setting
class Config:
    def __init__(self,mode='train'):
        #Path Setting
        self.test_img_path='pre_data/test'
        self.img_path = 'pre_data/train'
        self.checkpoint='../../checkpoints'
        self.inp_size = (256,256)
        self.grid = (16,16) #inp//4
        self.cls_num = 20
        self.anchor_num = 6
        self.res = 50
        self.sizes = [(128,128),(192,192),(256,256),(320,320),(384,384),(448,448),(512,512)]
        self.nms_threshold = 0.5
        self.dc_threshold = 0.01
        self.anchors = [[0.18414403,0.30230376],[0.57590277,0.38793478],[0.7645372,0.79610235],[0.30366338,0.63555215],[0.07356203,0.12814362]] #(w,h),normalized
        self.bs =1
        if mode=='train':
            self.file='pre_data/train.json'
            self.bs = 24 # batch size
            #augmentation parameter
            #self.rot = 10
            #self.scale = 0.25
            self.crop = 0.2
            self.flip = True
            self.valid_scale = 0.25
            self.sigmas =[(1,1),(3,3),(5,5),(9,9)]
            #train_setting
            self.lr = 0.1
            self.weight_decay=5e-4
            self.min_lr = 5e-5
            self.lr_factor = 0.01
            #exp_setting
            self.save_every_k_epoch = 10

        elif mode=='val':
            self.file = 'pre_data/val.json'
            self.bs = 1
        elif mode=='trainval':
            self.file = 'pre_data/train_subset.json'
        elif mode=='test':
            self.file = 'pre_data/test.json'
        
