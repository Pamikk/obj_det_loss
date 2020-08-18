###libs
import os
import argparse
import torch
from torch.utils.data import DataLoader
###files
from config import Config as cfg
from dataProcessing import VOC_dataset as dataset
from models.network import *
from trainer import Trainer
import warnings
from loss_funcs import LossAPI

warnings.filterwarnings('ignore')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--resume", type=int, default=0, help="start from epoch?")
    parser.add_argument("--exp",type=str,default='exp',help="name of exp")
    parser.add_argument("--res",type=int,default=50,help="resnet depth")
    parser.add_argument("--val",type=bool,default=False,help="only validation")
    args = parser.parse_args()
    
    

    #get data config
    config  = cfg()
    val_cfg = cfg('val')
    trainval_cfg = cfg('trainval')
    train_set = dataset(config)
    val_set = dataset(val_cfg,mode='val')
    trainval_set = dataset(trainval_cfg,mode='val')
    train_loader = DataLoader(train_set,batch_size=config.bs,shuffle=True,pin_memory=False,collate_fn=train_set.collate_fn)
    val_loader = DataLoader(val_set,batch_size=val_cfg.bs,shuffle=False,pin_memory=False,collate_fn=val_set.collate_fn)
    trainval_loader = DataLoader(trainval_set,batch_size=trainval_cfg.bs,shuffle=False,pin_memory=False,collate_fn=val_set.collate_fn)
    datasets = {'train':train_loader,'val':val_loader,'trainval':trainval_loader}
    config.exp_name = args.exp
    config.device = torch.device("cuda")
    torch.cuda.empty_cache()
    config.res = args.res
    #network
    network = YOLO(config.anchor_num,config.cls_num)
    loss = LossAPI(config)

    det = Trainer(config,datasets,network,loss,(args.resume,args.epochs))
    if args.val:
        det.validate(det.start-1,mode='val')
        det.validate(det.start-1,mode='train')
    else:
        det.train()