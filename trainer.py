import torch
import torch.nn as nn
import time
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os
import json

from utils import Logger
from utils import non_maximum_supression as nms
from utils import cal_tp_per_item,ap_per_class
tosave=['mAP']
thresholds = [0.5,0.75,0.95]
class Trainer:
    def __init__(self,cfg,datasets,net,loss,epoch):
        self.cfg = cfg
        if 'train' in datasets:
            self.trainset = datasets['train']
            self.valset = datasets['val']
        if 'trainval' in datasets:
            self.trainval = datasets['trainval']
        else:
            self.trainval = False
        if 'test' in datasets:
            self.testset = datasets['test']
        self.net = net
        name = cfg.exp_name
        self.name = name
        self.checkpoints = os.path.join(cfg.checkpoint,name)
        self.device = cfg.device
        self.net = self.net
        self.optimizer = optim.Adam(self.net.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
        self.lr_sheudler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min', factor=cfg.lr_factor, threshold=0.0001,patience=cfg.patience,min_lr=cfg.min_lr)
        if not(os.path.exists(self.checkpoints)):
            os.mkdir(self.checkpoints)
        self.predictions = os.path.join(self.checkpoints,'pred')
        if not(os.path.exists(self.predictions)):
            os.mkdir(self.predictions)
        start,total = epoch
        self.start = start        
        self.total = total
        self.loss = loss
        log_dir = os.path.join(self.checkpoints,'logs')
        if not(os.path.exists(log_dir)):
            os.mkdir(log_dir)
        self.logger = Logger(log_dir)
        torch.cuda.empty_cache()
        self.save_every_k_epoch = cfg.save_every_k_epoch #-1 for not save and validate
        self.val_every_k_epoch = cfg.val_every_k_epoch
        self.upadte_grad_every_k_batch = 1

        self.best_mAP = 0
        self.best_mAP_epoch = 0

        self.movingAvg = 0
        self.bestMovingAvg = 0
        self.bestMovingAvgEpoch = 1e9
        self.early_stop_epochs = 50
        self.alpha = 0.95 #for update moving Avg
        self.nms_threshold = cfg.nms_threshold
        self.conf_threshold = cfg.dc_threshold
        self.save_pred = False
        self.adjust_lr = cfg.adjust_lr
        #load from epoch if required
        if start>0:
            self.load_epoch(str(start))
        if start==-1:
            self.load_last_epoch()
        if start==-2:
            self.load_epoch('best')#best moving
        self.net = self.net.to(self.device)
    def load_last_epoch(self):
        files = os.listdir(self.checkpoints)
        idx =0
        for name in files:
            if name[-3]=='.pt':
                idx = max(idx,int(name[6:-3]))
        if idx==0:
            return
        else:
            self.load_epoch(idx)
    def save_epoch(self,idx,epoch):
        saveDict = {'net':self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler':self.lr_sheudler.state_dict(),
                    'epoch':epoch,
                    'mAP':self.best_mAP,
                    'mAP_epoch':self.best_mAP_epoch,
                    'movingAvg':self.movingAvg,
                    'bestmovingAvg':self.bestMovingAvg,
                    'bestmovingAvgEpoch':self.bestMovingAvgEpoch}
        path = os.path.join(self.checkpoints,'epoch_'+idx+'.pt')
        torch.save(saveDict,path)                  
    def load_epoch(self,idx):
        model_path = os.path.join(self.checkpoints,'epoch_'+idx+'.pt')
        if os.path.exists(model_path):
            print('load:'+model_path)
            info = torch.load(model_path)
            self.net.load_state_dict(info['net'])
            if not(self.adjust_lr):
                self.optimizer.load_state_dict(info['optimizer'])#might have bugs about device
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
                self.lr_sheudler.load_state_dict(info['lr_scheduler'])
            self.start = info['epoch']+1
            self.best_mAP = info['mAP']
            self.best_mAP_epoch = info['mAP_epoch']
            self.movingAvg = info['movingAvg']
            self.bestMovingAvg = info['bestmovingAvg']
            self.bestMovingAvgEpoch = info['bestmovingAvgEpoch']
        else:
            print('no such model at:',model_path)
            exit()
    def _updateMetrics(self,mAP,epoch):
        if self.movingAvg ==0:
            self.movingAvg = mAP
        else:
            self.movingAvg = self.movingAvg * self.alpha + mAP*(1-self.alpha)
        if self.bestMovingAvg<self.movingAvg:
            self.bestMovingAvg = self.movingAvg
            self.bestMovingAvgEpoch = epoch
            self.save_epoch('bestm',epoch)
    def logMemoryUsage(self, additionalString=""):
        if torch.cuda.is_available():
            print(additionalString + "Memory {:.0f}Mb max, {:.0f}Mb current".format(
                torch.cuda.max_memory_allocated() / 1024 / 1024, torch.cuda.memory_allocated() / 1024 / 1024))

    def train_one_epoch(self):
        running_loss ={'xy':0.0,'wh':0.0,'conf':0.0,'cls':0.0,'obj':0.0,'all':0.0,'iou':0.0,'gou':0.0}
        self.net.train()
        n = len(self.trainset)
        for i,data in tqdm(enumerate(self.trainset)):
            inputs,labels = data
            outs = self.net(inputs.to(self.device).float())
            labels = labels.to(self.device).float()
            size = inputs.shape[-2:]
            display,loss = self.loss(outs,labels,size)
            del inputs,outs,labels
            for k in running_loss:
                if k in display.keys():
                    running_loss[k] += display[k]/n
            loss.backward()
            #solve gradient explosion problem caused by large learning rate or small batch size
            #nn.utils.clip_grad_value_(self.net.parameters(), clip_value=2) 
            nn.utils.clip_grad_norm_(self.net.parameters(),max_norm=2.0)
            if i == n-1 or (i+1) % self.upadte_grad_every_k_batch == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            del loss
        self.logMemoryUsage()
        return running_loss
    def train(self):
        print("strat train:",self.name)
        print("start from epoch:",self.start)
        print("=============================")
        self.optimizer.zero_grad()
        print(self.optimizer.param_groups[0]['lr'])
        epoch = self.start
        stop_epochs = 0
        #torch.autograd.set_detect_anomaly(True)
        while epoch < self.total and stop_epochs<self.early_stop_epochs:
            running_loss = self.train_one_epoch()            
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.write_loss(epoch,running_loss,lr)
            #step lr
            self.lr_sheudler.step(running_loss['all'])
            lr_ = self.optimizer.param_groups[0]['lr']
            if lr_ == self.cfg.min_lr:
                print(lr_-self.cfg.min_lr)
                stop_epochs +=1
            if lr_ != lr:
                self.save_epoch(str(epoch),epoch)
            if (epoch+1)%self.save_every_k_epoch==0:
                self.save_epoch(str(epoch),epoch)
            if (epoch+1)%self.val_every_k_epoch==0:                
                metrics = self.validate(epoch,'val',self.save_pred)
                self.logger.write_metrics(epoch,metrics,tosave)
                mAP = metrics['mAP']
                self._updateMetrics(mAP,epoch)
                if mAP >= self.best_mAP:
                    self.best_mAP = mAP
                    self.best_mAP_epoch = epoch
                    self.save_epoch('best',epoch)
                print("best so far with:",self.best_mAP)
                if self.trainval:
                    metrics = self.validate(epoch,'train',self.save_pred)
                    self.logger.write_metrics(epoch,metrics,tosave,mode='Trainval')
            epoch +=1
                
        print("Best mAP: {:.4f} at epoch {}".format(self.best_mAP, self.best_mAP_epoch))
        self.save_epoch(str(epoch-1),epoch-1)
    def validate(self,epoch,mode,save=False):
        self.net.eval()
        res = {}
        print('start Validation Epoch:',epoch)
        if mode=='val':
            valset = self.valset
        else:
            valset = self.trainval
        with torch.no_grad():
            mAP = 0
            count = 0
            batch_metrics = dict.fromkeys(thresholds,[])
            gt_labels = []
            for _,data in tqdm(enumerate(valset)):
                inputs,labels,info = data
                outs = self.net(inputs.to(self.device).float())
                size = inputs.shape[-2:]
                pds = self.loss(outs,size=size,infer=True)
                nB = pds.shape[0]
                gt_labels += labels[:,1].tolist()               
                for b in range(nB):
                    pred = pds[b].view(-1,self.cfg.cls_num+5)
                    if save:
                        pds_ = list(pred.cpu().numpy().astype(float))
                        pds_ = [list(pd) for pd in pds_]
                        result = [pds_,pad]
                        res[name] = result
                    pred_nms = nms(pred,self.conf_threshold, self.nms_threshold)
                    if pred_nms.shape[0]>0:
                       ##print(pred_nms[0])
                    name = info['img_id'][b]
                    size = info['size'][b]
                    pad = info['pad'][b]
                    ##print(name)
                    ##print(pad)
                    gt = labels[labels[:,0]==b,1:].reshape(-1,5)                   
                    pred_nms[:,:4]*=size
                    pred_nms[:,0] -= pad[1]
                    pred_nms[:,1] -= pad[0]
                    if pred_nms.shape[0]>0:
                       ##print(pred_nms[0])
                    count+=1
                    for th in batch_metrics:
                        batch_metrics[th].append(cal_tp_per_item(pred_nms,gt,th))
                ##exit()
        metrics = {}
        for th in batch_metrics:
            tps,scores,pd_labels = [np.concatenate(x, 0) for x in list(zip(*batch_metrics[th]))]
            precision, recall, AP,_,_ = ap_per_class(tps, scores, pd_labels, gt_labels)
            mAP += np.mean(AP)
            metrics['AP/'+str(th)] = np.mean(AP)
            metrics['Precision/'+str(th)] = np.mean(precision)
            metrics['Recall/'+str(th)] = np.mean(recall)
        metrics['mAP'] = mAP/len(thresholds)
        if save:
            json.dump(res,open(os.path.join(self.predictions,'pred_epoch_'+str(epoch)+'.json'),'w'))
        
        return metrics
    def test(self):
        self.net.eval()
        res = {}
        with torch.no_grad():
            for _,data in tqdm(enumerate(self.testset)):
                inputs,info = data
                outs = self.net(inputs.to(self.device).float())
                size = inputs.shape[-2:]
                pds = self.loss(outs,size=size,infer=True)
                nB = pds.shape[0]
                for b in range(nB):
                    pred = pds[b].view(-1,self.cfg.cls_num+5)
                    name = info['img_id'][b]
                    size = info['size'][b]
                    pad = info['pad'][b]
                    pred[:,:4]*=size
                    pred[:,0] -= pad[1]
                    pred[:,1] -= pad[0]                    
                    pred_nms = nms(pred,self.conf_threshold, self.nms_threshold)
                    pds_ = list(pred_nms.cpu().numpy().astype(float))
                    pds_ = [list(pd) for pd in pds_]
                    res[name] = pds_
        
        json.dump(res,open(os.path.join(self.predictions,'pred_test.json'),'w'))

        


                


        




