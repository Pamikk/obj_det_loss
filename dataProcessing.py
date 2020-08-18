import torch.utils.data as data
import torch
import json
import numpy as np
import random
import cv2
import os
from torch.nn import functional as F


#stack functions for collate_fn
#Notice: all dicts need have same keys and all lists should have same length
def stack_dicts(dicts):
    if len(dicts)==0:
        return None
    res = {}
    for k in dicts[0].keys():
        res[k] = [obj[k] for obj in dicts]
    return res

def stack_list(lists):
    if len(lists)==0:
        return None
    res = list(range(len(lists[0])))
    for k in range(len(lists[0])):
        res[k] = torch.stack([obj[k] for obj in lists])
    return res

def brightness_scale(src,vs):
    img = cv2.cvtColor(src,cv2.COLOR_RGB2HSV).astype(np.float)
    img[:,:,2] *= (1+vs)
    img[:,:,2][img[:,:,2]>255] = 255
    return img

def augment(src,ang,vs,flip=False):
    #flip
    if flip:
        dst = cv2.flip(src,1)
    else:
        dst = src
    #rotation
    h,w,_ = dst.shape
    center =(w/2,h/2)
    mat = cv2.getRotationMatrix2D(center, ang, 1.0)
    dst = cv2.warpAffine(dst,mat,(w,h))  
    return brightness_scale(dst,vs),mat
def resize(src,tsize):
    dst = cv2.resize(src,(tsize[1],tsize[0]),interpolation=cv2.INTER_LINEAR)
    return dst    

def color_normalize(img,mean):
    img = img.astype(np.float)
    if img.max()>1:
        img /= 255
    img -= np.array(mean)/255
    return img

class VOC_dataset(data.Dataset):
    def __init__(self,cfg,mode='train'):
        self.img_path = cfg.img_path
        self.cfg = cfg
        data = json.load(open(cfg.file,'r'))
        self.imgs = list(data.keys())
        self.annos = data
        self.mode = mode
    def __len__(self):
        return len(self.imgs)

    def img_to_tensor(self,img):
        data = torch.tensor(np.transpose(img,[2,0,1]),dtype=torch.float)
        data /= data.max()
        return data
    def gen_gts(self,anno,pad=(0,0)):
        gts = torch.zeros((anno['obj_num'],5),dtype=torch.float)
        if anno['obj_num'] == 0:
            return gts
        bboxs = anno['annotation']
        for i in range(anno['obj_num']):
            gts[i,0] = bboxs[i]['label']
            x1,y1,x2,y2 = bboxs[i]['bbox']
            gts[i,1:] =torch.tensor([(x1+x2)/2+pad[1]-1,(y1+y2)/2+pad[0]-1,x2-x1,y2-y1],dtype=torch.float)
        return gts
        
    def get_trans_gts(self,labels,size,mat=np.eye(3),flip=True):
        #transfer
        if len(labels)== 0:
            return labels
        h,w = size
        cos = abs(mat[0,0])
        sin = abs(mat[0,1])
        xs = labels[:,1].clone()
        ys = labels[:,2].clone()
        ws = labels[:,3].clone()
        hs = labels[:,4].clone()
        if flip:
            xs = w-1-xs
        
        sy = 1/h #normalize to [0,1]
        sx = 1/w
        n = len(labels)

        pts = np.stack([xs,ys,np.ones([n])],axis=1).T
        tpts = torch.tensor(np.dot(mat,pts).T)
        labels[:,1] = tpts[:,0]*sx
        labels[:,2] = tpts[:,1]*sy
        labels[:,3] = (cos*ws + sin*hs)*sx
        labels[:,4] = (cos*hs + sin*ws)*sy
        return labels

    def pad_to_square(self,img):
        h,w,_= img.shape
        diff = abs(h-w)
        if w>h:
            pad = (diff//2,0,diff-diff//2,0)
        else:
            pad = (0,diff//2,0,diff-diff//2)
        img = cv2.copyMakeBorder(img,pad[0],pad[2],pad[1],pad[3],cv2.BORDER_CONSTANT,0)
        return img,(pad[0],pad[1])

    def __getitem__(self,idx):
        name = self.imgs[idx]
        anno = self.annos[name]
        img = cv2.imread(os.path.join(self.img_path,name+'.jpg'))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h,w,_ = img.shape
        if h!=w:
            img,pad = self.pad_to_square(img)
        else:
            pad = (0,0)
        h,w,_ = img.shape
        labels = self.gen_gts(anno,pad)
        if self.mode=='train':
            if random.uniform(0,1)>=0.25:
                rot = random.uniform(-1,1)*self.cfg.rot
            else:
                rot = 0
            if random.uniform(0,1)>=0.5:
                flip = True
            else:
                flip = False
            if random.uniform(0,1)>=0.5:
                vs = self.cfg.valid_scale*random.uniform(-1,1)
            else:
                vs = 0
            dst,mat = augment(img,rot,vs,flip)
            labels = self.get_trans_gts(labels,(h,w),mat,flip)#normalize to[0,1]
            data = self.img_to_tensor(dst)
            #labels = self.fill_with_zeros(labels,n)
            return data,labels        
        else:
            #validation set
            tsize = (448,448)# (h//64*64,w//64*64)
            data = resize(img,tsize)
            data = self.img_to_tensor(data)
            info ={'size':h,'img_id':name,'pad':pad}
            if self.mode=='val':
                return data,labels,info
            else:
                return data,info
    def collate_fn(self,batch):
        if self.mode=='test':
            data,info = list(zip(*batch))
            data = torch.stack(data)
            info = stack_dicts(info)
            return data,info 
        elif self.mode=='val':
            data,labels,info = list(zip(*batch))
            info = stack_dicts(info)
            data = torch.stack(data)
        elif self.mode=='train':
            data,labels = list(zip(*batch))
            tsize = random.sample(self.cfg.sizes,1)[0]
            data = torch.stack([F.interpolate(img.unsqueeze(0),tsize,mode='bilinear',align_corners=True).squeeze(0) for img in data])    
        tmp =[]
                   
                
        for i,bboxes in enumerate(labels):
            if len(bboxes)>0:
                label = torch.zeros(len(bboxes),6)
                label[:,1:] = bboxes
                label[:,0] = i
                tmp.append(label)
        if len(tmp)>0:
            labels = torch.cat(tmp,dim=0)
            labels = labels.reshape(-1,6)
        else:
            labels = torch.tensor(tmp,dtype=torch.float)
        if self.mode=='train':
            return data,labels
        else:
            return data,labels,info

                





