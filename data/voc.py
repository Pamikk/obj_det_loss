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
    conts =[]
    if root is None:
        return conts
    for cont in root:
        conts.append(dtype(cont.text))
    return conts
def get_one_anno(path):
    tree = ET.parse(path)
    root = tree.getroot()
    anno={}
    anno['img_name'] = root.find('filename').text
    size = root.find('size')
    anno['size'] = get_all_cont(size,int)
    objs = root.findall('object')
    anno['obj_num'] = len(objs)
    objects = []
    for obj in objs:
        item={}
        name=obj.find('name').text
        label = voc_classes[name]-1#ignore background
        assert label>=0
        bbox= obj.find('bndbox')
        bbox = get_all_cont(bbox,float)
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
def split_annotation(anno,split_path):
    test_list = open(os.path.join(split_path,'test.txt'),'r')
    train_list = open(os.path.join(split_path,'train.txt'),'r')
    val_list = open(os.path.join(split_path,'val.txt'),'r')
    test = {}
    train = {}
    val = {}
    trainval={}
    for name in test_list.readlines():
        name = name.strip()
        test[name] = anno[name]
    count = 0
    for name in train_list.readlines():
        name = name.strip()
        train[name] = anno[name]
        if count<200:
            trainval[name] = anno[name]
        count+=1 
    for name in val_list.readlines():
        name = name.strip()
        val[name] = anno[name]
    print(len(test),len(train),len(val))
    return test,train,val,trainval  

anno_path = '../../dataset/VOCdevkit/VOC2007/Annotations'
annos = get_annotations(anno_path)
json.dump(annos,open('annotation_voc07.json','w'))
#annos = json.load(open('annotation_voc07.json','r'))

split_path = '../../dataset/VOCdevkit/VOC2007/ImageSets/Main'
test,train,val,trainval = split_annotation(annos,split_path)
json.dump(test,open('test.json','w'))
json.dump(train,open('train.json','w'))
json.dump(val,open('val.json','w'))
json.dump(trainval,open('trainval.json','w'))


