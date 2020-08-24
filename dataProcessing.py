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
def rotate(src,ang,labels):
    h,w,_ = src.shape
    center =(w/2,h/2)
    mat = cv2.getRotationMatrix2D(center, ang, 1.0)
    dst = cv2.warpAffine(src,mat,(w,h))
    labels_ = labels.clone()
    xs,ys,ws,hs = labels[:,1:].T
    n = len(xs)
    cos = abs(mat[0,0])
    sin = abs(mat[0,1])
    pts = np.stack([xs,ys,np.ones([n])],axis=1).T
    tpts = torch.tensor(np.dot(mat,pts).T,dtype=torch.float)
    labels_[:,1] = tpts[:,0]
    labels_[:,2] = tpts[:,1]
    labels_[:,3] = (cos*ws + sin*hs)
    labels_[:,4] = (cos*hs + sin*ws)
    return dst,labels_
def flip(src,labels):
    w = src.shape[1]
    dst = cv2.flip(src,1)
    labels[:,1] = w-1-labels[:,1]
    return dst,labels
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
        data /= 255.0
        return data
    def gen_gts(self,anno):
        gts = torch.zeros((anno['obj_num'],5),dtype=torch.float)
        if anno['obj_num'] == 0:
            return gts
        bboxs = anno['annotation']
        for i in range(anno['obj_num']):
            gts[i,0] = bboxs[i]['label']
            x1,y1,x2,y2 = bboxs[i]['bbox']
            gts[i,1:] =torch.tensor([(x1+x2)/2-1,(y1+y2)/2-1,x2-x1,y2-y1],dtype=torch.float)
        return gts
        
    def normalize_gts(self,labels,size):
        #transfer
        if len(labels)== 0:
            return labels
        labels[:,1:]/=size 
        return labels

    def pad_to_square(self,img):
        h,w,_= img.shape
        ts = (max(h,w)//32+1)*32
        diff2 = abs(w-ts)
        pad = (diff1//2,diff2//2,diff1-diff1//2,diff2-diff2//2)
        img = cv2.copyMakeBorder(img,pad[0],pad[2],pad[1],pad[3],cv2.BORDER_CONSTANT,0)
        return img,(pad[0],pad[1])

    def __getitem__(self,idx):
        name = self.imgs[idx]
        anno = self.annos[name]
        img = cv2.imread(os.path.join(self.img_path,name+'.jpg'))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img,pad = self.pad_to_square(img)
        h = img.shape[0]
        labels = self.gen_gts(anno)
        if self.mode=='train':
            labels[:,1]+=pad[1]
            labels[:,2]+=pad[0]
            if random.randint(0,1)==1:
                img,labels = flip(img,labels)
            data = self.img_to_tensor(img)
            labels = self.normalize_gts(labels,h)
            #labels = self.fill_with_zeros(labels,n)
            return data,labels      
        else:
            #validation set
            data = self.img_to_tensor(img)
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
            #ratio = random.choice(ratios)
            scale = random.choice(self.cfg.sizes)
            tsize = (scale,scale)
            data = torch.stack([F.interpolate(img.unsqueeze(0),tsize,mode='bilinear',align_corners=True).squeeze(0) for img in data]) #multi-scale-training   
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

                





