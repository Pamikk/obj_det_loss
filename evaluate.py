import numpy as np
import json
import os
import torch
from tqdm import tqdm

from utils import non_maximum_supression_eval as nms
from utils import cal_tp_per_item,ap_per_class
from config import Config
def gen_gts(anno):
    gts = torch.zeros((anno['obj_num'],5),dtype=torch.float)
    if anno['obj_num'] == 0:
        return gts
    bboxs = anno['annotation']
    for i in range(anno['obj_num']):
        gts[i,0] = bboxs[i]['label']
        x1,y1,x2,y2 = bboxs[i]['bbox']
        gts[i,1:] =torch.tensor([(x1+x2)/2-1,(y1+y2)/2-1,x2-x1,y2-y1],dtype=torch.float)
    return gts
cfg = Config(mode='trainval')
gts = json.load(open(cfg.file))
nms_threshold = 0.3
conf_threshold = 0
tosave = ['mAP']
plot = [0.5,0.75] 
thresholds = np.around(np.arange(0.5,0.96,0.05),2)
pds = json.load(open(os.path.join(cfg.checkpoint,'debug','pred','pred_test.json')))
mAP = 0
count = 0
batch_metrics={}
for th in thresholds:
    batch_metrics[th] = []
gt_labels = []
for i,img in tqdm(enumerate(gts.keys())):
    pred_nms = torch.tensor(pds[img])
    gt = gen_gts(gts[img])
    gt_labels += gt[:,0].tolist()  
    #pred_nms = nms(pred,conf_threshold, nms_threshold)
    total = 0
    for th in batch_metrics:
        batch_metrics[th].append(cal_tp_per_item(pred_nms,gt,th))
    exit()
metrics = {}
for th in batch_metrics:
    tps,scores,pd_labels = [np.concatenate(x, 0) for x in list(zip(*batch_metrics[th]))]
    precision, recall, AP,_,_ = ap_per_class(tps, scores, pd_labels, gt_labels)
    mAP += np.mean(AP)
    if th in plot:
        metrics['AP/'+str(th)] = np.mean(AP)
        metrics['Precision/'+str(th)] = np.mean(precision)
        metrics['Recall/'+str(th)] = np.mean(recall)
metrics['mAP'] = mAP/len(thresholds)
for k in metrics:
    print(k,':',metrics[k])
