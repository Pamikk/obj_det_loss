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
        count = np.zeros(size)
        for anno in annos[name]['annotation']:
            xmin,ymin,xmax,ymax = anno['bbox']
            xmin,ymin,xmax,ymax = int(xmin-1),int(ymin-1),int(xmax-1),int(ymax-1)
            count[ymin:ymax,xmin:xmax]+=1
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
            bw /= w
            bh /= h
            allb.append((bw,bh))
            mh = min(mh,bh)
            mxh = max(mxh,bh)
            mw = min(mw,bw)
            mxw = max(mxw,bw)
    km = kmeans(allb,k=5,max_iters=500)
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
        ts = round(w/64)*64/(round(h/64)*64)
        if ts in res.keys():
            res[ts]+=1
        else:
            res[ts] = 1
        ts = round(max(h,w)/64)*64
        if ts in res2.keys():
            res2[ts]+= 1/len(annos)
        else:
            res2[ts] = 1/len(annos)

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
#[0.18414403,0.30230376] 7576
#[0.57590277,0.38793478] 2893
#[0.7645372,0.79610235] 4402
#[0.30366338,0.63555215] 4393
#[0.07356203,0.12814362] 8931

