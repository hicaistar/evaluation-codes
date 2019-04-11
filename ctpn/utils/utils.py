import numpy as np 
import xmltodict 
import os
import cv2
import matplotlib.pyplot as plt 
import json
import tensorflow as tf 
from keras.utils import to_categorical
import keras
import csv

anchor_scale = 16
#
IOU_NEGATIVE =0.3
IOU_POSITIVE = 0.8
IOU_POSITIVE_NAME = 0.8
IOU_SELECT =0.7 

RPN_POSITIVE_NUM=350
RPN_TOTAL_NUM=600

#bgr  can find from  here https://github.com/fchollet/deep-learning-models/blob/master/imagenet_utils.py
IMAGE_MEAN = [123.68,116.779,103.939]

DEBUG = True


    
def readtxt(path,with_label=False,label_dict =None,img_font='jpg'):
    '''
    load annotation from the text file
    :param path:
    :return:
    '''
    gtboxes = []
    name_index = []
    if not os.path.exists(path):
        return np.array(text_polys, dtype=np.float32)
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            label = line[-1]
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
            if len(line)==9:
                x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
                gtboxes.append((int(x1), int(y1), int(x3), int(y3)))
            elif len(line) ==5:
                x1,y1,x2,y2 =  list(map(float, line[:4]))
                gtboxes.append((int(x1), int(y1), int(x2), int(y2)))
            if with_label and label_dict is not None:
                name_index.append(label_dict(label))
    imgfile = os.path.basename(path).replace('txt',img_font)
    if with_label:
        return np.array(gtboxes),imgfile,name_index
    else:
        return np.array(gtboxes),imgfile


def readxml(path,with_label=False,label_dict =None):
    gtboxes=[]
    name_index = []
    imgfile = ''
    with open(path,'rb') as f :
        xml = xmltodict.parse(f)
        bboxes = xml['annotation']['object']
        if(type(bboxes)!=list):
            x1 = bboxes['bndbox']['xmin']
            y1 = bboxes['bndbox']['ymin']
            x2 = bboxes['bndbox']['xmax']
            y2 = bboxes['bndbox']['ymax']
            gtboxes.append((int(x1),int(y1),int(x2),int(y2)))
            if with_label and label_dict is not None:
                name_index.append(label_dict[bboxes['name']])
        else:
            for i in bboxes:
                x1 = i['bndbox']['xmin']
                y1 = i['bndbox']['ymin']
                x2 = i['bndbox']['xmax']
                y2 = i['bndbox']['ymax']
                gtboxes.append((int(x1),int(y1),int(x2),int(y2)))
                if with_label and label_dict is not None:
                    name_index.append(label_dict[i['name']])
        imgfile = xml['annotation']['filename']
    if with_label:
        return np.array(gtboxes),imgfile,name_index
    else:
        return np.array(gtboxes),imgfile

def gen_anchor(featuresize,scale):

    """
    gen base anchor from feature map [HXW][10][4]
    reshape  [HXW][10][4] to [HXWX10][4]
    """
    
    heights=[11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    widths=[16,16,16,16,16,16,16,16,16,16]

    #gen k=9 anchor size (h,w)
    heights = np.array(heights).reshape(len(heights),1)
    widths = np.array(widths).reshape(len(widths),1)
  
    base_anchor = np.array([0,0,15,15])
    #center x,y
    xt = (base_anchor[0] + base_anchor[2]) * 0.5 
    yt = (base_anchor[1] + base_anchor[3]) * 0.5

    # x1 y1 x2 y2 给出anchor的大概位置s
    x1 = xt - widths * 0.5
    y1 = yt - heights * 0.5 
    x2 = xt + widths * 0.5
    y2 = yt + heights * 0.5  
    # base_anchor.shape 为(10,4)
    base_anchor = np.hstack((x1,y1,x2,y2))
    
    h,w = featuresize
    shift_x = np.arange(0,w) * scale
    shift_y = np.arange(0,h) * scale
    #apply shift
    anchor = []
    # anchor 在feature_map上逐个pixel移动的时候，在原图上是怎么移动，返回原图上anchor的坐标
    for i in shift_y:
        for j in shift_x:
            anchor.append( base_anchor + [j,i,j,i])          
    return np.array(anchor).reshape((-1,4))


def cal_iou(box1,box1_area, boxes2,boxes2_area):
    """
    calculate the IoU(Intersection over Union) between
    box1 [x1,y1,x2,y2]
    boxes2 [Msample,x1,y1,x2,y2]
    """
    x1 = np.maximum(box1[0],boxes2[:,0])
    x2 = np.minimum(box1[2],boxes2[:,2])
    y1 = np.maximum(box1[1],boxes2[:,1])
    y2 = np.minimum(box1[3],boxes2[:,3])

    intersection = np.maximum(x2-x1,0) * np.maximum(y2-y1,0)
    iou = intersection / (box1_area + boxes2_area[:] - intersection[:])
    return iou

def cal_overlaps(boxes1,boxes2):
    """
    boxes1 [Nsample,x1,y1,x2,y2]  anchor
    boxes2 [Msample,x1,y1,x2,y2]  grouth-box
    
    """
    area1 = (boxes1[:,0] - boxes1[:,2]) * (boxes1[:,1] - boxes1[:,3])
    area2 = (boxes2[:,0] - boxes2[:,2]) * (boxes2[:,1] - boxes2[:,3])

    overlaps = np.zeros((boxes1.shape[0],boxes2.shape[0]))

    #calculate the intersection of  boxes1(anchor) and boxes2(GT box)
    for i in range(boxes1.shape[0]):
        overlaps[i][:] = cal_iou(boxes1[i],area1[i],boxes2,area2)
    
    return overlaps


def bbox_transfrom(anchors,gtboxes):
    """
     compute relative predicted vertical coordinates Vc ,Vh
     with respect to the bounding box location of an anchor 
     return:
         an array contains the parameters which as the values for regression 
    """
    regr = np.zeros((anchors.shape[0],2))
    Cy = (gtboxes[:,1] + gtboxes[:,3]) * 0.5
    Cya = (anchors[:,1] + anchors[:,3]) * 0.5
    h = gtboxes[:,3] - gtboxes[:,1] + 1.0
    ha = anchors[:,3] - anchors[:,1] + 1.0

    Vc = (Cy - Cya) / ha
    Vh = np.log(h/ha)
    return np.vstack((Vc,Vh)).transpose()


def bbox_transfor_inv(anchor,regr):
    """
        from anchor parameter return predict bbox
        return:
            bbox: the bbox after regression 
    """
    
    Cya = (anchor[:,1] + anchor[:,3]) * 0.5
    ha = anchor[:,3] - anchor[:,1] + 1 

    Vcx = regr[0,:,0]
    Vhx = regr[0,:,1]

    Cyx = Vcx * ha + Cya
    hx = np.exp(Vhx) * ha
    xt = (anchor[:,0] + anchor[:,2]) * 0.5

    x1 = xt - 16 * 0.5
    y1 = Cyx - hx * 0.5 
    x2 = xt + 16 * 0.5
    y2 = Cyx + hx * 0.5  
    bbox = np.vstack((x1,y1,x2,y2)).transpose()
    
    return bbox


def clip_box(bbox,im_shape):
    ''' clip the box if the area not in the image
        return:
            bbox: clipped box (x1,y1,x2,y2)
    '''
    
    # x1 >= 0
    bbox[:, 0] = np.maximum(np.minimum(bbox[:, 0], im_shape[1] - 1), 0)
    # y1 >= 0
    bbox[:, 1] = np.maximum(np.minimum(bbox[:, 1], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    bbox[:, 2] = np.maximum(np.minimum(bbox[:, 2], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    bbox[:, 3] = np.maximum(np.minimum(bbox[:, 3], im_shape[0] - 1), 0)

    return bbox


def filter_bbox(bbox,minsize):
    '''filter the bboxes through the size of the bbox
        return:
            keep: the index of the bbox which satisfy the condition of the size
    '''
    ws = bbox[:, 2] - bbox[:, 0] + 1
    hs = bbox[:, 3] - bbox[:, 1] + 1
    keep = np.where((ws >= minsize) & (hs >= minsize))[0]
    return keep





def get_session(gpu_fraction=0.6):  
    '''''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''  
  
    num_threads = os.environ.get('OMP_NUM_THREADS')  
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)  
  
    if num_threads:  
        return tf.Session(config=tf.ConfigProto(  
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))  
    else:  
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  
  
class random_uniform_num():
    """
    uniform random
    """
    def __init__(self,total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0
    def get(self,batchsize):
        r_n=[]
        if(self.index+batchsize>self.total):
            r_n_1 = self.range[self.index:self.total]
            np.random.shuffle(self.range)
            self.index = (self.index+batchsize)-self.total
            r_n_2 = self.range[0:self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)
            
        else:
            r_n = self.range[self.index:self.index+batchsize]
            self.index = self.index+batchsize
        return r_n


def cal_rpn(imgsize, base_anchor,gtboxes):
    ''' calculate the parameters and its label for bounding boxes
        parameters:
            imgsize: the size of input image
            base_anchor: a list of anchor for this size of image
            gtboxes: the grouth true boxes in the image
        return:
            [labels: the label for the bbox classify text/non-text
             bbox_targets: the parameters(height,y_center) for bbox 
            ]
            base_anchor: a list of anchor for this size of image

    '''
    #       (h,w) , (h/16,w/16),16,   gtbox)  ground_truth box
    imgh,imgw = imgsize

    #gen base anchor 有9个anchor了

    #calculate iou
    overlaps = cal_overlaps(base_anchor,gtboxes)
    #init labels -1 don't care  0 is negative  1 is positive
    labels = np.empty(base_anchor.shape[0])
    labels.fill(-1)

    #for each GT box corresponds to an anchor which has highest IOU 
    gt_argmax_overlaps = overlaps.argmax(axis=0)

    #the anchor with the highest IOU overlap with a GT box
    anchor_argmax_overlaps = overlaps.argmax(axis=1)
    anchor_max_overlaps = overlaps[range(overlaps.shape[0]),anchor_argmax_overlaps] 



    #IOU > IOU_POSITIVE
    labels[anchor_max_overlaps>IOU_POSITIVE]=1
    #IOU <IOU_NEGATIVE
    labels[anchor_max_overlaps<IOU_NEGATIVE]=0
    #ensure that every GT box has at least one positive RPN region
    labels[gt_argmax_overlaps] = 1

    #only keep anchors inside the image
    outside_anchor = np.where(
       (base_anchor[:,0]<0) |
       (base_anchor[:,1]<0) |
       (base_anchor[:,2]>=imgw)|
       (base_anchor[:,3]>=imgh)
       )[0]
    labels[outside_anchor]=-1

    #subsample positive labels ,if greater than RPN_POSITIVE_NUM(default 128)
    fg_index = np.where(labels==1)[0]
    if(len(fg_index)>RPN_POSITIVE_NUM):
        labels[np.random.choice(fg_index,len(fg_index)-RPN_POSITIVE_NUM,replace=False)]=-1

    #subsample negative labels 
    bg_index = np.where(labels==0)[0]
    num_bg = RPN_TOTAL_NUM - np.sum(labels==1)
    if(len(bg_index)>num_bg):
        labels[np.random.choice(bg_index,len(bg_index)-num_bg,replace=False)]=-1


    # calculate bbox targets
    bbox_targets = bbox_transfrom(base_anchor,gtboxes[anchor_argmax_overlaps,:])

    return [labels,bbox_targets],base_anchor


def cal_rpn_with_name(imgsize, base_anchor,gtboxes,name_index):
    ''' calculate the parameters and its label for bounding boxes
        parameters:
            imgsize: the size of input image
            base_anchor: a list of anchor for this size of image
            gtboxes: the grouth true boxes in the image
        return:
            [labels: the label for the bbox classify text/non-text
             bbox_targets: the parameters(height,y_center) for bbox 
             name_label: the labels for each text bboxes 
            ]
            base_anchor: a list of anchor for this size of image                
    '''

    #       (h,w) , (h/16,w/16),16,   gtbox)  ground_truth box
    imgh,imgw = imgsize

    #gen base anchor 有9个anchor了

    #calculate iou
    overlaps = cal_overlaps(base_anchor,gtboxes)
    #init labels -1 don't care  0 is negative  1 is positive
    labels = np.empty(base_anchor.shape[0])
    labels.fill(-1)

    #for each GT box corresponds to an anchor which has highest IOU 
    gt_argmax_overlaps = overlaps.argmax(axis=0)

    #the anchor with the highest IOU overlap with a GT box
    anchor_argmax_overlaps = overlaps.argmax(axis=1)
    anchor_max_overlaps = overlaps[range(overlaps.shape[0]),anchor_argmax_overlaps] 



    #IOU > IOU_POSITIVE
    labels[anchor_max_overlaps>IOU_POSITIVE]=1
    #IOU <IOU_NEGATIVE
    labels[anchor_max_overlaps<IOU_NEGATIVE]=0
    #ensure that every GT box has at least one positive RPN region
    labels[gt_argmax_overlaps] = 1


    # add an array for the bbox labels
    name_label = np.empty(base_anchor.shape[0])
    name_label.fill(-1)
    name_index = np.array(name_index)
    name_index = name_index[anchor_argmax_overlaps]
    #IOU > IOU_POSITIVE
    name_label[anchor_max_overlaps>IOU_POSITIVE_NAME]=name_index[anchor_max_overlaps>IOU_POSITIVE_NAME]
#     #IOU <IOU_NEGATIVE
    name_label[anchor_max_overlaps<IOU_NEGATIVE]=0
        #ensure that every GT box has at least one positive RPN region
    name_label[gt_argmax_overlaps] = name_index[gt_argmax_overlaps]
#     one_hot_label = to_categorical(name_label,num_classes=16) #n_classes = ,0,name_class


    #only keep anchors inside the image
    outside_anchor = np.where(
       (base_anchor[:,0]<0) |
       (base_anchor[:,1]<0) |
       (base_anchor[:,2]>=imgw)|
       (base_anchor[:,3]>=imgh)
       )[0]
    labels[outside_anchor]=-1
    name_label[outside_anchor]=-1

    #subsample positive labels ,if greater than RPN_POSITIVE_NUM(default 128)
    fg_index = np.where(labels==1)[0]
    if(len(fg_index)>RPN_POSITIVE_NUM):
        labels[np.random.choice(fg_index,len(fg_index)-RPN_POSITIVE_NUM,replace=False)]=-1

    #subsample negative labels 
    bg_index = np.where(labels==0)[0]
    num_bg = RPN_TOTAL_NUM - np.sum(labels==1)
    if(len(bg_index)>num_bg):
        #print('bgindex:',len(bg_index),'num_bg',num_bg)
        labels[np.random.choice(bg_index,len(bg_index)-num_bg,replace=False)]=-1


    #subsample positive name labels ,if greater than RPN_POSITIVE_NUM(default 128)
    fg_index = np.where(name_label>0)[0]
    if(len(fg_index)>RPN_POSITIVE_NUM):
        name_label[np.random.choice(fg_index,len(fg_index)-RPN_POSITIVE_NUM,replace=False)]=-1

    #subsample negative labels 
    bg_index = np.where(name_label==0)[0]
    num_bg = RPN_TOTAL_NUM - np.sum(name_label>0)
    if(len(bg_index)>num_bg):
        #print('bgindex:',len(bg_index),'num_bg',num_bg)
        name_label[np.random.choice(bg_index,len(bg_index)-num_bg,replace=False)]=-1

    # calculate bbox targets
    bbox_targets = bbox_transfrom(base_anchor,gtboxes[anchor_argmax_overlaps,:])

    return [labels,bbox_targets,name_label],base_anchor


class CTPN_DataGenerator(keras.utils.Sequence):
    def __init__(self,image_dir,label_dir,label_format='VOCdevkit',
                 batch_size=1,dim=None,shuffle=True,label_dict=None,img_font='jpg'):
        ''' ctpn datagenerator 
        parameters:
            image_dir: the path of the dir contains images which has text region;
            label_dir: the corresponding path of the label describe the boundingboxes info;
            label_format: support 'icdar','VOCdevkit'; 
            batch_size: batch size generator generated, only work when dim is set;
            dim: the input images dimension, only accept rgb images;
            shuffle: shuffle the samples on the end of each epochs or not;
            label_dict: the bounding box label dict; 
        '''
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.dim = dim
        self.shuffle = shuffle
        self.anchor_scale = 16
        self.label_format = label_format
        self.img_font = img_font
        if label_format == 'VOCdevkit':
            self.label_files = [label_file for label_file in os.listdir(self.label_dir) \
                                  if os.path.splitext(label_file)[1] == '.xml']
        elif label_format == 'icdar':
            self.label_files = [label_file for label_file in os.listdir(self.label_dir) \
                                  if os.path.splitext(label_file)[1] == '.txt']
        if self.dim:
            assert dim[2] == 3
            self.base_anchor = gen_anchor((int(dim[0]/self.anchor_scale),int(dim[1]/self.anchor_scale)),self.anchor_scale)
            self.anchor_len = len(self.base_anchor)
            self.batch_size = batch_size
        else:
            print('Warning: no dimension provided, batch size is set to 1 ')
            self.batch_size = 1
        self.label_dict=label_dict
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.label_files)) / self.batch_size)
    
    def __getitem__(self,index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find list of IDs
        list_label_files_temp = [self.label_files[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_label_files_temp)
        return X,y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.label_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
        
    def __data_generation(self,list_label_files_temp):
        ''' Generates data containing batch_size samples
            return: 
                X: the prepared input image array(0 mean)
                {'rpn_class_reshape': the label for classify text/non-text
                 'rpn_regress_reshape': the parameters(height,y_center) for bbox 
                 'rpn_name_cls_reshape': the label for each bbox 
                 }
        ''' 
        if self.dim:
            X = np.empty((self.batch_size,*self.dim),dtype=np.float32)
            cls_array = np.empty((self.batch_size,1,self.anchor_len),dtype=np.float32)
            regr_array = np.empty((self.batch_size,self.anchor_len,3),dtype=np.float32)
            name_cls_array = np.empty((self.batch_size,1,self.anchor_len),dtype=np.float32)
        
        for i,label_file in enumerate(list_label_files_temp):
            try:
                if self.label_format == 'VOCdevkit':
                    if self.label_dict:
                        gtbox,imgfile,name_index = readxml(self.label_dir + "/" + label_file,with_label=True,label_dict =self.label_dict)
                    else:
                        gtbox,imgfile = readxml(self.label_dir + "/" + label_file)
                elif self.label_format == 'icdar':
                    if self.label_dict:
                        gtbox,imgfile,name_index = readtxt(self.label_dir + "/" + label_file,with_label=True,label_dict = self.label_dict,img_font=self.img_font)
                    else:
                        gtbox,imgfile = readtxt(self.label_dir + "/" + label_file,
                                                img_font=self.img_font)
                img_path = os.path.join(self.image_dir,imgfile)
#                 print(img_path)
                img = cv2.imread(img_path)
                
            except (OSError,IOError) as error:
                print(error)
                
            m_img = img - IMAGE_MEAN

            if self.dim:
                h,w = self.dim[:2]
                base_anchor = self.base_anchor
                X[i,] = m_img
            else:
                h,w = img.shape[:2]
                base_anchor = gen_anchor((int(h/self.anchor_scale),int(w/self.anchor_scale)),self.anchor_scale)
                X = np.expand_dims(m_img,axis=0)
                anchor_len = len(base_anchor)
                cls_array = np.empty((self.batch_size,1,anchor_len),dtype=np.float32)
                regr_array = np.empty((self.batch_size,anchor_len,3),dtype=np.float32)
                name_cls_array = np.empty((self.batch_size,1,anchor_len),dtype=np.float32)
                
            if self.label_dict:
                [cls,regr,name_cls],_ = cal_rpn_with_name((h,w),base_anchor,gtbox,name_index=name_index)
            else:
                [cls,regr],_ = cal_rpn((h,w),base_anchor,gtbox)
            #zero-center by mean pixel 
#                 m_img = np.expand_dims(m_img,axis=0)

            # 这里是指在regr的最后1维加text 判断
            regr = np.hstack([cls.reshape(cls.shape[0],1),regr])
#                 name_cls = np.hstack([cls.reshape(cls.shape[0],1),
#                                       name_cls.reshape(name_cls.shape[0],1)])

            cls = np.expand_dims(cls,axis=0)
            cls_array[i,:,:] = cls
            #regr = np.expand_dims(regr,axis=1)
#                 regr = np.expand_dims(regr,axis=0)
            regr_array[i,:,:] = regr
            if self.label_dict:
                name_cls =  np.expand_dims(name_cls,axis=0)
                name_cls_array[i,:,:] = name_cls
                return X,{'rpn_class_reshape':cls_array,
                              'rpn_regress_reshape':regr_array,
                              'rpn_name_cls_reshape':name_cls_array} 
            else:
                return X,{'rpn_class_reshape':cls_array,
                              'rpn_regress_reshape':regr_array,}
            
            

    
        
def nms(dets, thresh):
    ''' non maximum supression'''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

