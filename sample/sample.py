import os
import numpy as np
import requests
import operator
import json
import argparse
import time
from PIL import Image

def metric():
    lasttime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    result = {
    "accuracy":"0.9",
    "updatetime": lasttime
    }
    return result

def load_data(data_dir):
    print("data dir:",data_dir)
    files = os.listdir(data_dir)
    print("get files:",files)
    images = []
    for file in files:
        if os.path.splitext(file)[1] == '.png':
            images.append(file)
    first = images[0]
    image = Image.open(os.path.join(data_dir, first))
    img = image.resize((224,224))
    raw_data = np.array(img)/255.0

    # Normalization
    data = raw_data - np.array([0.485,0.456,0.406])
    data = np.divide(data, np.array([0.229,0.224,0.225]))
    data = np.moveaxis(data, 2, 0)
    data = np.expand_dims(data, 0)
    return data

def main(data_dir,output,name,server):

    data = load_data(data_dir)
    # send request
    url = ('http://%s/predict' % server)
    try:
        r = requests.post(url,
            data = json.dumps({"instances":[{"gpu_0/data_0": data.astype(np.float32).tolist()}]}))
    except Exception:
        print("serving error.")
    else:
        if r.status_code != 200:
            print("status:",r.content)
            result = {
            "name":name,
            "result":"error"
            }
        else:
            result_json = json.loads(r.text)
            result = result_json['predictions'] if 'predictions' in result_json else result_json['outputs']
            idx, _ = max(enumerate(result[0][0]), key = operator.itemgetter(1))
            print("idx:",idx)
            res = ("%s" % metric())
            result = {
            "name":name,
            "result":res,
            }
    # write result
    filename = ("%s.json" % name)
    file = os.path.join(output,filename)
    with open(file, 'w') as f:
        json.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process args for evaluation')
    parser.add_argument("--output", "-o", help="set output directory")
    parser.add_argument("--data", "-d", help="set dataset directory")
    parser.add_argument("--name","-n", help="set function name")
    parser.add_argument("--server","-s", help="set serving server address")
    args = parser.parse_args()

    main(args.data,args.output,args.name,args.server)
