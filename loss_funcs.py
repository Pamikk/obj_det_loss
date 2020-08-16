import torch.nn as nn
import torch
import numpy as np
from utils import iou_wo_center,iou_wt_center,gou,cal_gous
#directly get by normalized anchor size, not accute according to YOLOv2, need change distance metric
__all__=["MyLoss","MyLoss_v2","MyLoss_v3","MyLoss_v4"]  
#Functional Utils
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()
def dice_loss1d(pd,gt,threshold=0.5):
    assert pd.shape == gt.shape
    if gt.shape[0]==0:
        return 0
    inter = torch.sum(pd*gt)
    pd_area = torch.sum(torch.pow(pd,2))
    gt_area = torch.sum(torch.pow(gt,2))
    dice = (2*inter+1)/(pd_area+gt_area+1)
    #fix nans
    dice[dice != dice] = dice.new_tensor([1.0])
    return 1-dice.mean()
def dice_loss(pd,gt,threshold=0.5):
    dims = tuple(range(len(pd.shape)))
    inter = torch.sum(pd*gt,dim=dims)
    pd_area = torch.sum(torch.pow(pd,2),dim=dims)
    gt_area = torch.sum(torch.pow(gt,2),dim=dims)
    dice = (2*inter+1)/(pd_area+gt_area+1)
    #fix nans
    dice[dice != dice] = dice.new_tensor([1.0])
    return 1-dice.mean()

def make_grid_mesh(grid_size,device='cuda'):
    x = np.arange(0,grid_size,1)
    y = np.arange(0,grid_size,1)
    grid_x,grid_y = np.meshgrid(x,y)
    grid_x = torch.tensor(grid_x).view(1,1,grid_size,grid_size).to(dtype=torch.float,device=device)
    grid_y = torch.tensor(grid_y).view(1,1,grid_size,grid_size).to(dtype=torch.float,device=device)
    return grid_x,grid_y
class LossAPI(nn.Module):
    def __init__(self,cfg):
        super(LossAPI,self).__init__()
        self.bbox_loss = YOLOLossv3(cfg)
    def forward(self,out,gt=None,infer=False):
        if infer:
            return self.bbox_loss(out,infer=infer)
        else:
            ret,loss = self.bbox_loss(out,gt)
            return ret,loss

class YOLOv1Loss(nn.Module):
    #revised YOLOv1 Loss,noanchorbased
    def __init__(self,bnum=5,cls_num=20):
        super(YOLOv1Loss,self).__init__()
        self.object_scale = 5
        self.noobject_scale = 0.5
        self.cls_num = cls_num
        self.ignore_thres = 0.5
        self.device= 'cuda'
        self.num_anchor = bnum
        self.channel_num = bnum*(self.cls_num+5)
       
    def build_target(self,pd,labels,th):
        self.device ='cuda' if pd.is_cuda else 'cpu'
        nB,nA,nG,_,_ = pd.shape
        #threshold = th
        nGts = len(labels)
        #create output tensors
        obj_mask = torch.zeros(nB,nA,nG,nG,dtype=torch.bool,device=self.device)
        noobj_mask = torch.ones(nB,nA,nG,nG,dtype=torch.bool,device=self.device)
        #cls_mask = torch.zeros(nB,nA,nG,nG,dtype=torch.float,device=self.device)
        scores = torch.zeros(nB,nA,nG,nG,dtype=torch.float,device=self.device) #iou score
        tx = torch.zeros(nB,nA,nG,nG,dtype=torch.float,device=self.device)  
        ty = torch.zeros(nB,nA,nG,nG,dtype=torch.float,device=self.device) 
        tw = torch.zeros(nB,nA,nG,nG,dtype=torch.float,device=self.device) 
        th = torch.zeros(nB,nA,nG,nG,dtype=torch.float,device=self.device) 
        #tcls = torch.zeros(nB,nA,nG,nG,self.cls_num,dtype=torch.float,device=self.device) 
        if nGts==0:
            return obj_mask,noobj_mask,tx,ty,tw,th,obj_mask.float()
        #convert target
        gts = labels[:,1:]*nG
        gxs = gts[:,0]
        gys = gts[:,1]

        return obj_mask,noobj_mask,tx,ty,tw,th,obj_mask.float()
    def forward(self,out,gts,infer=False):
        nb,nc,nh,_ = out.shape
        self.device ='cuda' if out.is_cuda else 'cpu'
        nG = nh
        pred = out.view(nb,self.num_anchor,5,nG,nG).permute(0,1,3,4,2).contiguous()
        #reshape to nB,nA,nH,nW,bboxes
        xs = torch.sigmoid(pred[:,:,:,:,0])#dxs
        ys = torch.sigmoid(pred[:,:,:,:,1])#dys
        ws = pred[:,:,:,:,2]
        hs = pred[:,:,:,:,3]
        conf = torch.sigmoid(pred[:,:,:,:,4])#Object score
        

        if grid_size != self.grid_size:
            self.get_mesh_grid(grid_size)
            #self.get_anchors(grid_size)

        pd_bboxes = torch.zeros_like(pred[:,:,:,:,:4],dtype=torch.float,device=self.device)
        pd_bboxes[:,:,:,:,0] = (xs + self.grid_x)/grid_size
        pd_bboxes[:,:,:,:,1] = (ys + self.grid_y)/grid_size
        pd_bboxes[:,:,:,:,2] = ws
        pd_bboxes[:,:,:,:,3] = hs

        
        if infer:
            return torch.cat((pd_bboxes,conf.unsqueeze(dim=-1)),axis=-1)
        else:
            obj_mask,noobj_mask,tx,ty,tw,th,tconf = self.my_build_target(pd_bboxes,gts,self.ignore_thres)

        loss_x = mse_loss(xs[obj_mask],tx[obj_mask])
        loss_y = mse_loss(ys[obj_mask],ty[obj_mask])
        loss_xy = loss_x + loss_y
        loss_w = mse_loss(torch.log(ws[obj_mask]),torch.log(tw[obj_mask]))
        loss_h = mse_loss(torch.log(hs[obj_mask]),torch.log(th[obj_mask]))
        loss_wh = loss_w + loss_h
        #pds = torch.stack([xs[obj_mask],ys[obj_mask],ws[obj_mask],hs[obj_mask]],axis=-1) 
        #gts = torch.stack([tx[obj_mask],ty[obj_mask],tw[obj_mask],th[obj_mask]],axis=-1)  
        #ious = 1-iou_wo_center(ws[obj_mask],hs[obj_mask],tw[obj_mask],th[obj_mask])
        #loss_wh = (ious*ious).mean()
        loss_gou = torch.tensor(0.0,dtype=torch.float,device=self.device)
        loss_iou = torch.tensor(0.0,dtype=torch.float,device=self.device)
        loss_conf = torch.tensor(0.0,dtype=torch.float,device=self.device)
        for b in range(nb):
           idx = gts[:,0] == b
           if idx.sum()==0:
               loss_conf += conf[b].mean()/nb
               continue
           ious,gous = cal_gous(pd_bboxes[b].view(-1,4),gts[idx,1:])
           pd_gous,indices = gous.max(dim=1)
           n,m = gous.shape
           pd_ious = ious[range(n),indices]
           mask = pd_ious >= self.ignore_thres
           gt_gous,inds= gous.max(dim=0) #select max overlap
           loss_gou += (1 - gt_gous).mean()/nb #punish false negatives
           loss_iou += (1-ious[inds,range(m)]).mean()/nb
           mask_ = torch.zeros(n,dtype=mask.dtype,device=self.device)
           mask_[inds] = 1 #poses
           ignore = (~mask)|mask_  #over threshold but not the max,hard negative= mask & ~mask_
           #hard poses= (~mask)&mask_ #max but not greater than the threshold
           #negative = (~mask) & (~mask_)
           #easy poses = mask & mask_
           mask[inds] = 1
           loss_gou += (1-pd_gous[mask]).mean()/nb
           loss_iou += (1-pd_ious[mask]).mean()/nb
           tconfb = obj_mask[b].reshape(-1)[ignore]|mask_[ignore] #avoid positive conflict
           loss_conf += dice_loss1d(conf[b].reshape(-1)[ignore],tconfb.float())/nb #conf loss based on iou
           #ignore over ignore threshold but not the max overlap



        #loss_iou = torch.tensor(0)
        #loss_gou = torch.tensor(0)
        loss_conf = dice_loss(conf,tconf) + loss_conf #conf loss based on position

        #loss_cls = bce_loss(cls_conf[obj_mask],tcls[obj_mask])
        total_loss = loss_xy + loss_wh + loss_conf +loss_gou
        #add loss_xy,loss_wh to avoid gradient disappear
      
        return [loss_xy,loss_wh,loss_conf,loss_iou,loss_gou,total_loss]

class YOLOLossv3(nn.Module):
    def __init__(self,cfg=None):
        super(YOLOLossv3,self).__init__()
        self.object_scale = 1
        self.noobject_scale = 0.5
        self.cls_num = cfg.cls_num
        self.grid_size = 0
        self.ignore_thres = 0.5
        self.device= 'cuda'
        self.target_num = 120
        self.num_anchor = cfg.anchor_num
        self.anchors = np.array(cfg.anchors).reshape(-1,2)
        self.channel_num = self.num_anchor*(self.cls_num+5)
    def build_target(self,pds,gts,anchors):
        self.device ='cuda' if pd.is_cuda else 'cpu'
        nB,nA,nG,_,_ = pds.shape
        nC = self.cls_num
        #threshold = th
        nGts = len(gts)
        obj_mask = torch.zeros(nB,nA,nG,nG,dtype=torch.bool,device=self.device)
        noobj_mask = torch.ones(nB,nA,nG,nG,dtype=torch.bool,device=self.device)
        txs = torch.zeros(nB,nA,nG,nG,dtype=torch.float,device=self.device)  
        tys = torch.zeros(nB,nA,nG,nG,dtype=torch.float,device=self.device) 
        tws = torch.zeros(nB,nA,nG,nG,dtype=torch.float,device=self.device) 
        ths = torch.zeros(nB,nA,nG,nG,dtype=torch.float,device=self.device)
        tcls = torch.zeros(nB,nA,nG,nG,nC,dtype=torch.float,device=self.device) 
        if nGts==0:
            return obj_mask,noobj_mask,txs,tys,tws,ths,tcls,obj_mask.float()
        #convert target
        gt_boxes = gts[:,2:]*nG
        gws = gt_boxes[:,2]
        ghs = gt_boxes[:,3]
        anchors = torch.tensor(anchors*nG,device=self.device).reshape(-1,2)
        #calculate bbox ious with anchors
        ious = torch.stack([iou_wo_center(gws,ghs,anchor_w,anchor_h) for (anchor_w,anchor_h) in anchors])
        _, best_n = ious.max(0)

        batch,gt_labels = gts[:,:2].long()
        gxs,gys = gt_boxes[:,0],gt_boxes[:,1]
        gis,gjs = gxs.long(),gys.long()
        obj_mask[batch,best_n,gjs,gis] = 1
        noobj_mask[batch,best_n,gjs,gis] = 0
        #ignore big overlap but not the best
        for i,iou in enumerate(iou.t()):
            noobj_mask[batch[i],iou > self.ignore_thres,gjs[i],gis[i]] = 0
        
        txs[batch,best_n,gjs,gis] = gxs-gxs.floor()
        tys[batch,best_n,gjs,gis] = gys-gys.floor()
        tws[batch,best_n,gjs,gis] = torch.log(gws/anchors[best_n][:,0])
        tys[batch,best_n,gjs,gis] = torch.log(ghs/anchors[best_n][:,1])

        tcls[batch,best_n,gjs,gis,gt_labels] = 1
        return obj_mask,noobj_mask,txs,tys,tws,ths,tcls,obj_mask.float()


    def forward(self,out,gts=None,infer=False):
        nb,nc,nh,nw = out.shape
        self.device ='cuda' if out.is_cuda else 'cpu'
        grid_size = nh
        pred = out.view(nb,self.num_anchor,self.channel_num,nh,nw).permute(0,1,3,4,2).contiguous()
        #reshape to nB,nA,nH,nW,bboxes
        xs = torch.sigmoid(pred[...,0])#dxs
        ys = torch.sigmoid(pred[...,1])#dys
        ws = pred[...,2]
        hs = pred[...,3]
        conf = torch.sigmoid(pred[...,4])#Object score
        cls_score = torch.sigmoid(pred[...,5:])
        #grid,anchors
        grid_x,grid_y = make_grid_mesh(grid_size,self.device)
        anchors_w = torch.tensor(self.anchors[:,0]*grid_size).view((1, self.num_anchors, 1, 1))
        anchors_h = torch.tensor(self.anchors[:,1]*grid_size).view((1, self.num_anchors, 1, 1))

        pd_bboxes = torch.zeros_like(pred[:,:,:,:4],dtype=torch.float,device=self.device)
        pd_bboxes[:,:,:,0] = (xs + grid_x)/grid_size
        pd_bboxes[:,:,:,1] = (ys + grid_y)/grid_size
        pd_bboxes[:,:,:,2] = torch.exp(ws)*anchors_w
        pd_bboxes[:,:,:,3] = torch.exp(hs)*anchors_h

        if infer:
            return torch.cat((pred.view(nb,-1,4),conf.view(nb,-1,1),cls_score.view(nb,-1,self.cls_num)),dim=-1)
        else:
            return obj_mask,noobj_mask,txs,tys,tws,ths,tcls,tconf= self.build_target(pd_bboxes,gts)

        loss_x = mse_loss(xs[obj_mask],txs[obj_mask])
        loss_y = mse_loss(ys[obj_mask],tys[obj_mask])
        loss_xy = loss_x + loss_y
        loss_w = mse_loss(ws[obj_mask],tw[obj_mask])
        loss_h = mse_loss(hs[obj_mask],th[obj_mask])
        loss_wh = loss_w + loss_h

        loss_conf_obj = self.object_scale*bce_loss(conf[obj_mask],tconf[obj_mask])
        loss_conf_noobj = self.noobject_scale*bce_loss(conf[noobj_mask],tconf[noobj_mask])

        loss_conf = loss_conf_noobj+loss_conf_obj
        
        loss_cls = bce_loss(cls_score[obj_mask],tcls[obj_mask])

        total_loss =  loss_conf + loss_xy + loss_wh + loss_cls
        losses={'cls':loss_cls.item(),'obj':loss_conf_obj.item(),'noobj':loss_conf_noobj.item(),'conf':loss_conf.item(),'xy':loss_xy.item(),'wh':loss_wh.item()}
        return losses,total_loss
        
class myYOLOLoss(nn.Module):
    #anchor less
    #every cell is responsible for possible 
    #no need to build target
    #select gt based on ious
    def __init__(self):
        super(myYOLOLoss,self).__init__()
        #self.cls_num = 1#only wheat in this case,so ignore
        self.grid_size = 0
        self.ignore_thres = 0.5
        self.device= 'cuda'
    def get_mesh_grid(self,grid_size):
        self.grid_size = grid_size
        x = np.arange(0,grid_size,1)
        y = np.arange(0,grid_size,1)
        self.grid_x,self.grid_y = np.meshgrid(x,y)
        self.grid_x = torch.tensor(self.grid_x).view(1,1,grid_size,grid_size).to(dtype=torch.float,device=self.device)
        self.grid_y = torch.tensor(self.grid_y).view(1,1,grid_size,grid_size).to(dtype=torch.float,device=self.device)
    def get_gous_ious(self,pds,gts):
        nB = pds.shape[0]
        pds = pds.view(nB,-1,4)
        n = pds.shape[1]
        m = gts.shape[0]
        #num of pds for every batch
        mask = torch.zeros(nB,n,dtype=torch.bool,device=self.device)
        pd_gous = torch.zeros(nB,n,dtype=torch.float,device=self.device)
        pd_ious = torch.zeros(nB,n,dtype=torch.float,device=self.device)
        pd_indices = torch.zeros(nB,n,dtype=torch.long,device=self.device)
        if m==0:
            return pd_gous,pd_ious,mask,pd_indices    
        gt_indices = torch.tensor(range(m),dtype=torch.long,device=self.device)
        for b in range(nB):
            idx = gts[:,0]==b
            gt_n = idx.sum().item()
            if gt_n==0:
                continue
            pd_b = pds[b]
            gt_b = gts[idx,1:]
            ious,gous = cal_gous(pd_b,gt_b)
            #pd_n x gt_n
            gtb_idx = gt_indices[idx]#corresponding idx of groundtruth            
            selected = torch.zeros(gt_n,dtype=torch.bool,device=self.device)
            restm = gt_n
            assert gt_n < n
            while restm > 0:
                ious_,gous_ = ious[mask[b]==0,:][:,selected==0],gous[mask[b]==0,:][:,selected==0]
                #restn x restm
                vals_g,pd_idx = gous_.max(dim=0)
                #len:restm
                vals_i = ious_[pd_idx,range(restm)]                
                order = torch.argsort(vals_g)
                gt_idx =-1*torch.ones(ious_.shape[0],dtype=torch.long,device=self.device)
                #get corresponding gt_id for chosen pds,-1 as no gt matched

                gt_idx[pd_idx[order]]= order 
                #for same pd, the one with greater gou will repalce
                select_gt = gt_idx[gt_idx>-1]
                select_pd = pd_idx[select_gt]

                tmp = torch.where(mask[b]==0)[0][select_pd]

                pd_ious[b][tmp] = vals_i[select_gt]
                pd_gous[b][tmp] = vals_g[select_gt]
                pd_indices[b][tmp] = gtb_idx[selected==0][select_gt]
                
                #update
                selected[torch.where(selected==0)[0][select_gt]]=1
                mask[b][tmp]=1
                restm -= len(select_gt)
            pd = pd_b[mask[b]==0] #pds no matched gts
            ious_,gous_ = ious[mask[b]==0],gous[mask[b]==0]
            vals_g,gt_idx = gous_.max(dim=1)
            vals_i = ious_[range(n-gt_n),gt_idx]
            pd_ious[b][mask[b]==0] = vals_i
            pd_gous[b][mask[b]==0] = vals_g
            pd_indices[b][mask[b]==0] = gtb_idx[gt_idx]
        return pd_gous,pd_ious,mask,pd_indices
    def forward(self,out,gts,infer=False):
        nb,nc,nh,nw = out.shape
        self.num_anchor = nc//5
        self.device ='cuda' if out.is_cuda else 'cpu'
        grid_size = nh
        pred = out.view(nb,self.num_anchor,5,nh,nw).permute(0,1,3,4,2).contiguous()
        if grid_size != self.grid_size:
            self.get_mesh_grid(grid_size)
        #reshape to nB,nA,nH,nW,bboxes
        xs = (torch.tanh(pred[:,:,:,:,0]) + 0.5 + self.grid_x)/grid_size
        ys = (torch.tanh(pred[:,:,:,:,1]) + 0.5 + self.grid_y)/grid_size
        ws = torch.exp(-torch.pow(pred[:,:,:,:,2],2))#normalize to [0,1]
        hs = torch.exp(-torch.pow(pred[:,:,:,:,3],2))
        conf = torch.sigmoid(pred[:,:,:,:,4])#Object score
        pd_bboxes = torch.stack((xs,ys,ws,hs),dim=-1)

        
        if infer:
            return torch.cat((pd_bboxes,conf.unsqueeze(dim=-1)),axis=-1)
        else:
            pd_gous,pd_ious,tp_mask,indices = self.get_gous_ious(pd_bboxes,gts)
        threshold_mask = pd_ious > self.ignore_thres
        mask = threshold_mask | tp_mask
        ignore = (~threshold_mask) | tp_mask
        pds = pd_bboxes.view(nb,-1,4)[tp_mask]
        if gts.shape[0]>0:
            tgts = gts[indices[tp_mask],1:]
            loss_x = mse_loss(pds[:,0],tgts[:,0])
            loss_y = mse_loss(pds[:,1],tgts[:,1])
            loss_w = mse_loss(torch.log(pds[:,2]),torch.log(tgts[:,2]))
            loss_h = mse_loss(torch.log(pds[:,3]),torch.log(tgts[:,3]))
            loss_gou = 1-pd_gous[tp_mask].mean()
            loss_iou = 1-pd_ious[tp_mask].mean()
            loss_xy = loss_x + loss_y
            loss_wh = loss_w + loss_h
             
        else:
            loss_wh = torch.tensor(0.0,dtype=torch.float,device=self.device)
            loss_xy = torch.tensor(0.0,dtype=torch.float,device=self.device)
            loss_iou = torch.tensor(0.0,dtype=torch.float,device=self.device)
            loss_gou = torch.tensor(0.0,dtype=torch.float,device=self.device)
        loss_conf_bce = bce_loss(conf.view(nb,-1)[ignore],tp_mask[ignore].float())
        loss_conf_dice = dice_loss1d(conf.view(nb,-1)[ignore],tp_mask[ignore].float())
        loss_conf =  loss_conf_dice + loss_conf_bce
        #tconf = pd_ious[~ignore]
        #loss_conf += dice_loss1d(conf.view(nb,-1)[~ignore],tconf)
        #loss_cls = bce_loss(cls_conf[obj_mask],tcls[obj_mask])
        total_loss = loss_conf + 2*loss_gou + loss_wh + loss_xy
        #add loss_xy,loss_wh to avoid gradient disappear
      
        return [loss_xy,loss_wh,loss_conf,loss_iou,loss_gou,total_loss]
class YOLOv1Loss(nn.Module):
    #anchor less
    #every cell is responsible for possible 
    #no need to build target
    #select gt based on ious
    def __init__(self):
        super(YOLOv1Loss,self).__init__()
        #self.cls_num = 1#only wheat in this case,so ignore
        self.grid_size = 0
        self.ignore_thres = 0.5
        self.device= 'cuda'
    def get_mesh_grid(self,grid_size):
        self.grid_size = grid_size
        x = np.arange(0,grid_size,1)
        y = np.arange(0,grid_size,1)
        self.grid_x,self.grid_y = np.meshgrid(x,y)
        self.grid_x = torch.tensor(self.grid_x).view(1,1,grid_size,grid_size).to(dtype=torch.float,device=self.device)
        self.grid_y = torch.tensor(self.grid_y).view(1,1,grid_size,grid_size).to(dtype=torch.float,device=self.device)
    def build_target(self,pds,gts):
        nB,nA,nG = pds.shape[:3]
        n = pds.shape[1]
        m = gts.shape[0]
        #num of pds for every batch
        mask = torch.zeros(nB,n,dtype=torch.bool,device=self.device)
        pd_gous = torch.zeros(nB,n,dtype=torch.float,device=self.device)
        pd_ious = torch.zeros(nB,n,dtype=torch.float,device=self.device)
        pd_indices = torch.zeros(nB,n,dtype=torch.long,device=self.device)
        if m==0:
            return pd_gous,pd_ious,mask,pd_indices    
        gt_indices = torch.tensor(range(m),dtype=torch.long,device=self.device)
        for b in range(nB):
            idx = gts[:,0]==b
            gt_n = idx.sum().item()
            if gt_n==0:
                continue
            pd_b = pds[b]
            gt_b = gts[idx,1:]
            ious,gous = cal_gous(pd_b,gt_b)
            #pd_n x gt_n
            gtb_idx = gt_indices[idx]#corresponding idx of groundtruth            
            selected = torch.zeros(gt_n,dtype=torch.bool,device=self.device)
            restm = gt_n
            assert gt_n < n
            while restm > 0:
                ious_,gous_ = ious[mask[b]==0,:][:,selected==0],gous[mask[b]==0,:][:,selected==0]
                #restn x restm
                vals_g,pd_idx = gous_.max(dim=0)
                #len:restm
                vals_i = ious_[pd_idx,range(restm)]                
                order = torch.argsort(vals_g)
                gt_idx =-1*torch.ones(ious_.shape[0],dtype=torch.long,device=self.device)
                #get corresponding gt_id for chosen pds,-1 as no gt matched

                gt_idx[pd_idx[order]]= order 
                #for same pd, the one with greater gou will repalce
                select_gt = gt_idx[gt_idx>-1]
                select_pd = pd_idx[select_gt]

                tmp = mask[b]==0

                pd_ious[b][tmp][select_pd] = vals_i[select_gt]
                pd_gous[b][tmp][select_pd] = vals_g[select_gt]
                pd_indices[b][tmp][select_pd] = gtb_idx[selected==0][select_gt]
                
                #update
                selected[torch.where(selected==0)[0][select_gt]]=1
                mask[b][torch.where(tmp)[0][select_pd]]=1
                restm -= len(select_gt)
            pd = pd_b[mask[b]==0] #pds no matched gts
            ious_,gous_ = ious[mask[b]==0],gous[mask[b]==0]
            vals_g,gt_idx = gous_.max(dim=1)
            vals_i = ious_[range(n-gt_n),gt_idx]
            pd_ious[b][mask[b]==0] = vals_i
            pd_gous[b][mask[b]==0] = vals_g
            pd_indices[b][mask[b]==0] = gtb_idx[gt_idx]
        return pd_gous,pd_ious,mask,pd_indices
    def forward(self,out,gts,infer=False):
        nb,nc,nh,nw = out.shape
        self.num_anchor = nc//5
        self.device ='cuda' if out.is_cuda else 'cpu'
        grid_size = nh
        pred = out.view(nb,self.num_anchor,5,nh,nw).permute(0,1,3,4,2).contiguous()
        if grid_size != self.grid_size:
            self.get_mesh_grid(grid_size)
        #reshape to nB,nA,nH,nW,bboxes
        xs = (torch.sigmoid(pred[:,:,:,:,0]) + self.grid_x)/self.grid_size#dxs
        ys = (torch.sigmoid(pred[:,:,:,:,1]) + self.grid_y)/self.grid_size#dys can predict object out of cells
        ws = torch.exp(-torch.pow(pred[:,:,:,:,2],2))#normalize to [0,1]
        hs = torch.exp(-torch.pow(pred[:,:,:,:,3],2))
        conf = torch.sigmoid(pred[:,:,:,:,4])#Object score
        pd_bboxes = torch.stack((xs,ys,ws,hs),dim=-1)

        
        if infer:
            return torch.cat((pd_bboxes,conf.unsqueeze(dim=-1)),axis=-1)
        else:
            pd_gous,pd_ious,tp_mask,indices = self.build_target(pd_bboxes,gts)
        threshold_mask = pd_ious > self.ignore_thres
        mask = threshold_mask | tp_mask
        ignore = (~threshold_mask) | tp_mask
        pds = pd_bboxes.view(nb,-1,4)[tp_mask]
        if gts.shape[0]>0:
            tgts = gts[indices[tp_mask],1:]
            loss_x = mse_loss(pds[:,0],tgts[:,0])
            loss_y = mse_loss(pds[:,1],tgts[:,1])
            loss_w = mse_loss(pds[:,2],tgts[:,2])
            loss_h = mse_loss(pds[:,3],tgts[:,3])
            loss_gou = 1-pd_gous[tp_mask].mean()
            loss_iou = 1-pd_ious[tp_mask].mean()
            loss_xy = loss_x + loss_y
            loss_wh = loss_w + loss_h
             
        else:
            loss_wh = torch.tensor(0.0,dtype=torch.float,device=self.device)
            loss_xy = torch.tensor(0.0,dtype=torch.float,device=self.device)
            loss_iou = torch.tensor(0.0,dtype=torch.float,device=self.device)
            loss_gou = torch.tensor(0.0,dtype=torch.float,device=self.device)
        loss_conf = dice_loss1d(conf.view(nb,-1)[ignore],tp_mask[ignore].float())
        #loss_cls = bce_loss(cls_conf[obj_mask],tcls[obj_mask])
        total_loss = loss_xy + loss_wh + loss_conf + loss_gou + loss_iou
        #add loss_xy,loss_wh to avoid gradient disappear
      
        return [loss_xy,loss_wh,loss_conf,loss_iou,loss_gou,total_loss]





        







        