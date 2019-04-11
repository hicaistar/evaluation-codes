import os
import json
import argparse
import requests
import numpy as np

# user codes

import sys
import csv
from PIL import Image

import utils
import text_proposal_connector as text_connect
import metrics
from text_detector_api import TextLineDetectorClient,Textimage_Generator

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def textline_extract(image,prediction,threshold=0.3):
    h,w,_ = image.shape
    cls  = np.array(prediction[0])
    regr = np.array(prediction[1])
    cls_prod = np.array(prediction[2])
    anchor = utils.gen_anchor((int(h/16),int(w/16)),16)
    bbox = utils.bbox_transfor_inv(anchor,regr)
    bbox = utils.clip_box(bbox,[h,w])
    #score > 0.7

    fg = np.where(cls_prod[0,:,1]>threshold)[0]
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
    text = list(text.astype('int32'))
    return text

def evaluate(text,gt_text):
    pred_text = [txt_item[:8] for txt_item in text]
    iou_matrix = metrics.match_IOU(pred_text,gt_text)
    precision = metrics.get_precision(iou_matrix,threshold=0.7)
    recall = metrics.get_recall(iou_matrix,threshold=0.7)
    return precision,recall

# implement of evaluation
def execute_evaluation(evaluation):
    # add user code here:
    # 1. data = load_data(evaluation.data)
    # 2. payload = {"instances":[{"input": data.astype(np.float32).tolist()}]}
    # 3. response = evaluation.post_request(payload)
    # 4. result = metrics(response)
    # 5. evaluation.write_output(result)
    generator = Textimage_Generator(data_dir=evaluation.data,txt_dir=evaluation.data)
    for data,label in generator:
        data = data - utils.IMAGE_MEAN
        p_data0 = {'input_2_1':np.expand_dims(data,axis=0).astype(np.float32).tolist()}
        param = {"instances":[p_data0]}
        predict_request = json.dumps(param)
        # response = requests.post(FLAGS.SERVER_URL, data=predict_request)
        prediction = evaluation.post_request(predict_request)

        prediction = np.array(prediction)
        text = textline_extract(data,prediction)
        precision2,recall2 = evaluate(text,label)
        break
    result = {
    'precision': precision2,
    'recall': recall2
    }
    evaluation.write_output(result)
    print('precision %04f,recall %04f'%(precision2,recall2))

# Do not change codes below
class Evaluation:
    def __init__(self, function, data, server, output):
        self.name = function
        self.server = server
        self.data = data
        self.output = output

    def post_request(self,payload):
        response = requests.post(self.server, payload)
        if response.status_code == 200:
            result_json = json.loads(response.text)
            result = result_json['predictions'] if 'predictions' in result_json else result_json['outputs']
            return result
        else:
            print("serving error:",response.text)
            return ''

    def write_output(self,result):
        res = json.dumps(result)
        out = {
            "name": self.name,
            "result":res,
        }
        filename = ("%s.json" % self.name)
        file = os.path.join(self.output,filename)
        with open(file, 'w') as f:
            json.dump(out, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process args for evaluation')
    parser.add_argument("--output", "-o", help="set output directory")
    parser.add_argument("--data", "-d", help="set dataset directory")
    parser.add_argument("--name","-n", help="set function name")
    parser.add_argument("--server","-s", help="set serving server address")
    args = parser.parse_args()

    server = "http://{s}/predict".format(s=args.server)

    evaluation = Evaluation(args.name, args.data, server, args.output)

    execute_evaluation(evaluation)