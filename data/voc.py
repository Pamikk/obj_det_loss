import os
import xml.etree.ElementTree as ET
import json
voc_classes= {'__background__':0, 'aeroplane':1, 'bicycle':2, 
          'bird':3, 'boat':4, 'bottle':5,'bus':6, 'car':7,
           'cat':8, 'chair':9,'cow':10, 'diningtable':11, 'dog':12,
            'horse':13,'motorbike':14, 'person':15, 'pottedplant':16,
            'sheep':17, 'sofa':18, 'train':19, 'tvmonitor':20}
def get_anno_files(rootdir):
    files={}
    dirs = os.listdir(rootdir)
    for i in dirs:
        path = os.path.join(rootdir,i)
        if os.path.isfile(path):
            name,ext= os.path.splitext(i)
            if ext == '.xml':
                files[name] = path
    return files
def get_all_cont(root,dtype):
    conts ={}
    vals = []
    if root is None:
        return conts
    for cont in root:
        conts[cont.tag] = dtype(cont.text)
        vals.append(dtype(cont.text))
    return conts,vals
def get_one_anno(path):
    tree = ET.parse(path)
    root = tree.getroot()
    anno={}
    anno['img_name'] = root.find('filename').text
    size = root.find('size')
    size,_ = get_all_cont(size,int)
    anno['size'] = [size['width'],size['height'],size['depth']]
    objs = root.findall('object')
    anno['obj_num'] = len(objs)
    objects = []
    for obj in objs:
        item={}
        name=obj.find('name').text
        label = voc_classes[name]-1#ignore background
        assert label>=0
        bbox = obj.find('bndbox')
        bbox,vals = get_all_cont(bbox,float)
        bbox = [bbox['xmin'],bbox['ymin'],bbox['xmax'],bbox['ymax']]
        hard = int(obj.find('difficult').text)
        item = [label]+bbox+[hard]
        objects.append(item)
    anno['labels'] = objects
    return anno
def get_annotations(anno_dir):
    files = get_anno_files(anno_dir)
    annos={}
    for idx in files:
        annos[idx] = get_one_anno(files[idx])
    return annos
def split_annotation(anno,split_path,mode,dataset):
    train_list = open(os.path.join(split_path,f'{mode}.txt'),'r')
    train = {}
    trainval={}
    count=0
    for name in train_list.readlines():
        name = name.strip()
        train[name] = anno[name]
        if count<200 and mode=='train':
            trainval[name] = anno[name]
        count+=1
    if  mode=='train':
        json.dump(trainval,open(f'trainval_{dataset}.json','w'))
    json.dump(train,open(f'{mode}_{dataset}.json','w'))
    print(mode,len(train)) 
dataset = 'VOC2012'
anno_path = f'../../dataset/VOCdevkit/{dataset}/Annotations'
annos = get_annotations(anno_path)
json.dump(annos,open(f'annotation_{dataset}.json','w'))
#annos = json.load(open(f'annotation_{dataset}.json','r'))

split_path = f'../../dataset/VOCdevkit/{dataset}/ImageSets/Main'
modes=['train','val']
for mode in modes:
    split_annotation(annos,split_path,mode,dataset)



