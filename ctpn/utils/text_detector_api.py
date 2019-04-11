import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import keras
import os
import glob
import sys
import cv2
import numpy as np
import time
import utils
import text_proposal_connector as text_connect
import metrics
import json
import requests
import csv
from model import get_model_with_labels,get_model

class TextLineDetector(object):
    def __init__(self,model_path=None,threshold=0.7):
        self.basemodel = get_model()
        self.basemodel.load_weights(model_path)
        self.threshold = threshold
    def textline_extract(self,image):
        assert len(image.shape) == 3 
        h,w,c= image.shape
        #zero-center by mean pixel 
        m_img = image - utils.IMAGE_MEAN

        m_img = np.expand_dims(m_img,axis=0)

        start = time.time()
        result = self.basemodel.predict(m_img)
        cls,regr,cls_prod  = result

        anchor = utils.gen_anchor((int(h/16),int(w/16)),16)

        bbox = utils.bbox_transfor_inv(anchor,regr)
        bbox = utils.clip_box(bbox,[h,w])

        #score > 0.7
        fg = np.where(cls_prod[0,:,1]>self.threshold)[0]
        select_anchor = bbox[fg,:]
        select_score = cls_prod[0,fg,1]
        select_anchor = select_anchor.astype('int32')

        #filter size
        keep_index = utils.filter_bbox(select_anchor,16)

        #nsm
        select_anchor = select_anchor[keep_index]
        select_score = select_score[keep_index]
        select_score = np.reshape(select_score,(select_score.shape[0],1))
        nmsbox = np.hstack((select_anchor,select_score))
        keep = utils.nms(nmsbox,0.3)
        select_anchor = select_anchor[keep]
        select_score = select_score[keep]

        #text line
        textConn = text_connect.TextProposalConnector()
        text = textConn.get_text_lines(select_anchor,select_score,[h,w])
        end = time.time()

        text = list(text.astype('int32'))
        
        return text

class Textimage_Generator(keras.utils.Sequence):
    ''' generator for EAST model'''
    def __init__(self,data_dir,txt_dir,shuffle=True):
        ''' data_dir: path images data
            txt_dir: path bounding box data
            batch_size: generate batch size
            background_ratio: the ratio for only backgroud images, which no bounding 
                              boxes in the samples 
        '''
        self.data_dir = data_dir
        self.txt_dir = txt_dir
        self.img_paths = self.get_images()
        self.shuffle = shuffle
        self.input_size = 512
        self.on_epoch_end()
        
    def __len__(self):
        return len(self.img_paths)
    
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.img_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_images(self):
        files = []
        for ext in ['jpg', 'png', 'jpeg', 'JPG']:
            files.extend(glob.glob(
                os.path.join(self.data_dir, '*.{}'.format(ext))))
        return files
    
    def load_annoataion(self,path):
        '''
        load annotation from the text file
        :param path:
        :return:
        '''
        text_polys = []
        text_tags = []
        if not os.path.exists(path):
            return np.array(text_polys, dtype=np.float32)
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                label = line[-1]
                # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
                line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]

                x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
                text_polys.append([ x1, y1, x2, y2, x3, y3, x4, y4])
                text_tags.append(label)
            return np.array(text_polys, dtype=np.float32), np.array(text_tags)
        
    def get_txt(self,img_path):
        txt_name = os.path.basename(img_path).replace(os.path.splitext(img_path)[-1], '.txt')
        txt_path = os.path.join(self.txt_dir,txt_name)
        return txt_path
    def __getitem__(self,index):
        index = self.indexes[index]
        # Find list of IDs
        img_path = self.img_paths[index]
        # Generate data
        img = cv2.imread(img_path)
#         img = img - utils.IMAGE_MEAN
        h, w, _ = img.shape
        txt_fn = self.get_txt(img_path)
#                 print(txt_fn)
        if not os.path.exists(txt_fn):
            raise IOError('the file %s is not exist'%s)
        text_polys, text_tags = self.load_annoataion(txt_fn)
        return img,text_polys   

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class TextLineDetectorClient(object):
    def __init__(self,server_url=None,threshold=0.7):
        self.server_url = server_url
        self.threshold = threshold
        img = np.random.rand(200,200,3)
        p_data0 = {'inputs_1':img}
        param = {"instances":[p_data0]}
        predict_request = json.dumps(param,cls=NumpyEncoder)
        response = requests.post(self.server_url, data=predict_request)
        response.raise_for_status()
        
    def textline_extract(self,image):
        assert len(image.shape) == 3 
        h,w,c= image.shape
        #zero-center by mean pixel 
        m_img = image - utils.IMAGE_MEAN
        
#         m_img = np.expand_dims(m_img,axis=0)
        p_data0 = {'inputs_1':m_img}
        param = {"instances":[p_data0]}
        predict_request = json.dumps(param,cls=NumpyEncoder)
        start = time.time()
        
        response = requests.post(self.server_url, data=predict_request)
        response.raise_for_status()
        prediction = response.json()['predictions'][0]
#         result = self.basemodel.predict(m_img)
        cls  = np.array(prediction['output0'])
        regr = np.array(prediction['output1'])
        cls_prod = np.array(prediction['output2'])
        cls = np.expand_dims(cls,axis=0)
        regr = np.expand_dims(regr,axis=0)
        cls_prod = np.expand_dims(cls_prod,axis=0)
        anchor = utils.gen_anchor((int(h/16),int(w/16)),16)

        bbox = utils.bbox_transfor_inv(anchor,regr)
        bbox = utils.clip_box(bbox,[h,w])

        #score > 0.7
        fg = np.where(cls_prod[0,:,1]>self.threshold)[0]
        select_anchor = bbox[fg,:]
        select_score = cls_prod[0,fg,1]
        select_anchor = select_anchor.astype('int32')

        #filter size
        keep_index = utils.filter_bbox(select_anchor,16)

        #nsm
        select_anchor = select_anchor[keep_index]
        select_score = select_score[keep_index]
        select_score = np.reshape(select_score,(select_score.shape[0],1))
        nmsbox = np.hstack((select_anchor,select_score))
        keep = utils.nms(nmsbox,0.3)
        select_anchor = select_anchor[keep]
        select_score = select_score[keep]

        #text line
        textConn = text_connect.TextProposalConnector()
        text = textConn.get_text_lines(select_anchor,select_score,[h,w])
        end = time.time()

        text = list(text.astype('int32'))
        
        return text
    
    def evaluate(self,image,gt_text):
        text = self.textline_extract(image)
        pred_text = [txt_item[:8] for txt_item in text]
        iou_matrix = metrics.match_IOU(pred_text,gt_text)
        precision = metrics.get_precision(iou_matrix)
        recall = metrics.get_recall(iou_matrix)
        return precision,recall

def draw_bbox(image,text):
    for i in text:
        cv2.line(image,(i[0],i[1]),(i[2],i[3]),(255,0,0),2)
        cv2.line(image,(i[0],i[1]),(i[4],i[5]),(255,0,0),2)
        cv2.line(image,(i[6],i[7]),(i[2],i[3]),(255,0,0),2)
        cv2.line(image,(i[4],i[5]),(i[6],i[7]),(255,0,0),2)
    return image