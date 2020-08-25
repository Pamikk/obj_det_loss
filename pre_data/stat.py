import json
import numpy as np
class kmeans(object):
    def __init__(self,vals,k=3,max_iters=200):
        self.vals = np.array(vals)
        print(self.vals.shape)
        self.dim = self.vals.shape[-1]
        self.k = k
        self.maxi = max_iters
        self.num = len(vals)
        self.terminate = False
    def initialization(self):
        assign = np.zeros(self.num,dtype=int)
        self.centers = list(range(self.k))
        k = self.k
        partn = self.num//k
        for i in range(k):
            assign[i*partn:(i+1)*partn] = i
        self.assign = assign
    def update_center(self):
        for i in range(self.k):
            if np.sum(self.assign==i)>0:
                #avoid empty cluster leading Error
                self.centers[i] = np.mean(self.vals[self.assign==i],axis=0)
        if type(self.centers) != np.ndarray:
            self.centers = np.array(self.centers)
    def cal_distance(self,obj1,obj2):
        #1-iou
        obj1 = obj1.reshape(-1,2)
        obj2 = obj2.reshape(-1,2)
        inter = np.minimum(obj1[:,0].reshape(-1,1),obj2[:,0].reshape(1,-1))*np.minimum(obj1[:,1].reshape(-1,1),obj2[:,1].reshape(1,-1))
        union = (obj1[:,0]*obj1[:,1]).reshape(-1,1) + (obj2[:,0]*obj2[:,1]).reshape(1,-1) - inter +1e-16
        return 1-inter/union
    def update_assign(self):
        self.terminate = True
        centers = self.centers
        for i in range(self.num):
            val = self.vals[i]
            tmp = self.cal_distance(val,np.stack(centers,axis=0))
            id = np.argmin(tmp)
            if id != self.assign[i]:
                self.assign[i] = id
                self.terminate= False
    def iter(self,num):
        self.update_center()
        self.update_assign()
        if self.terminate:
            self.print_cs()
            return
        else:
            if num == self.maxi:
                print("reach max iterations")
                self.print_cs()
                return
            else:
                return self.iter(num+1)
    def print_cs(self):
        for i in range(self.k):
            print(self.centers[i],np.sum(self.assign==i))
        print(self.cal_distance(self.centers,self.centers))
    def write2json(self,name):
        #assign = [[int(self.assign[i]),self.vals[i]] for i in range(self.num)]
        centers = list([[self.centers[i],np.sum(self.assign==i)] for i in range(self.k)])
        #json.dump(centers,open(name,'w'))
def count_overlap(annos):
    mc = 0
    for name in annos:
        size = annos[name]['size']
        w,h,_ = size
        count = np.zeros((16,16))
        for anno in annos[name]['annotation']:
            xmin,ymin,xmax,ymax = anno['bbox']
            t = max(w,h)
            xc = ((xmin+xmax)/2-1)/t
            yc = ((ymin+ymax)/2-1)/t
            count[int(16*yc),int(16*xc)]+=1
        mc = max(count.max(),mc)
    print(mc)
def analyze_hw(annos):
    allb = []
    mh,mw = 1,1
    mxh,mxw = 0,0
    for name in annos:
        size = annos[name]['size']
        w,h,_ = size
        for anno in annos[name]['annotation']:
            xmin,ymin,xmax,ymax = anno['bbox']
            bw,bh = xmax-xmin,ymax-ymin
            t = max(w,h)
            allb.append((bw/t,bh/t))
            mh = min(mh,bh)
            mxh = max(mxh,bh)
            mw = min(mw,bw)
            mxw = max(mxw,bw)
    km = kmeans(allb,k=4,max_iters=500)
    km.initialization()
    km.iter(0)  
    print(mh,mw,mxh,mxw)
def analyze_xy(annos):
    for name in annos:
        size = annos[name]['size']
        w,h,_ = size
        for anno in annos[name]['annotation']:
            xmin,ymin,xmax,ymax = anno['bbox']
            if ymax > h or xmax >w:
                print('???')
    print('finish')
def analyze_size(annos):
    res = {}
    res2 = {}
    for name in annos:
        size = annos[name]['size']
        w,h,_ = size
        ts = max(w,h)
        if ts in res.keys():
            res[ts]+=1
        else:
            res[ts] = 1
        ts = round(max(h,w)/16)*16
        if ts in res2.keys():
            res2[ts]+= 1/len(annos)
        else:
            res2[ts] = 1/len(annos)
    res2 = {k: v for k, v in sorted(res2.items(), key=lambda item: item[1])}
    print(res)
    print(len(res))
    print(res2)
    print(len(res2))

path ='annotation.json'

annos = json.load(open(path,'r'))
analyze_size(annos)
#img size:
#96 100 500 500
#overlap
#6
#center overlap 3(32,32)
#center overlap 4(16,16)


#[0.26533935 0.33382434] 10522
#[0.66550966 0.56042827] 7400
#[0.0880948  0.11774004] 12716

#[0.76822971 0.57259308] 4912
#[0.20632625 0.26720238] 8987
#[0.39598597 0.47268035] 5993
#[0.07779112 0.10330848] 10746

