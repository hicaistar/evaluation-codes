import os
import json
import argparse
import requests
import operator
import numpy as np
# tensorflow-1.12.0
import tensorflow as tf

# User codes
def metrics(y_predicted, y_ground_truth):
    assert len(y_predicted) == len(y_ground_truth)
    acc_num = 0
    for i in range(len(y_predicted)):
        result = y_predicted[i]
        idx, _ = max(enumerate(result[0][0]), key = operator.itemgetter(1))
        if idx == y_ground_truth[i]:
            acc_num = acc_num + 1
    accuracy = float(acc_num)/float(len(y_predicted))
    accuracy = str(accuracy)
    print("accuracy:",accuracy)
    return {'accuracy': accuracy}

def tf_parser(record):
    features = tf.parse_single_example(record,features={
    'image_raw':tf.FixedLenFeature([],tf.string),
    'label': tf.FixedLenFeature([],tf.int64),
    'height': tf.FixedLenFeature([],tf.int64),
    'width': tf.FixedLenFeature([],tf.int64),
    'channel': tf.FixedLenFeature([],tf.int64),
    })

    image = tf.decode_raw(features['image_raw'],tf.uint8)
    label = tf.cast(features['label'],tf.int32)
    height = tf.cast(features['height'],tf.int32)
    width = tf.cast(features['width'],tf.int32)
    channel = tf.cast(features['channel'],tf.int32)
    image = tf.reshape(image,[height,width,channel])
    image = tf.image.resize_images(image,[299,299],method=0)
    image = tf.cast(image,dtype=tf.float32)
    image = tf.multiply(image, 1/255.,)
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image,label

def load_data(data_dir, input_list, input_size_list, dtype_list):
    assert len(input_list) == len(input_size_list)
    assert len(input_list) == len(dtype_list)
    # tfrecord data
    filename = os.path.join(data_dir,'train_v1.tfrecord')
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(tf_parser)
    iterator = dataset.make_one_shot_iterator()
    x, label = iterator.get_next()

    with tf.Session() as sess:
        try:
            while True:
                input_data, ground_truth = sess.run([x, label])
                input_data = np.expand_dims(input_data, 0)
                feature = lambda s, t: input_data.astype(dtype=t).tolist()
                yield {
                    input_list[i]: feature(input_size_list[i], tf.as_dtype(dtype_list[i]).as_numpy_dtype)
                    for i in range(len(input_list))
                }, ground_truth
        except tf.errors.OutOfRangeError as e:
            print("finish all data.")

# Don't edit, provided by the platform.
class Evaluation:
    def __init__(self, name, data, server, output, manifest):
        self.name = name
        self.server = server
        self.data = data
        self.output = output
        self.manifest = manifest
        self.input_list = []
        self.input_size_list = []
        self.dtype_list = []

    def execute(self):
        self._parse_manifest()
        response_list = []
        ground_truth_list = []
        for feature, ground_truth in load_data(self.data, self.input_list,
                                               self.input_size_list,
                                               self.dtype_list):
            try:
                payload = json.dumps({"instances": [feature]})
                response = self._post_request(payload)
            except Exception as e:
                print(e)
                response_list.append(None)
            else:
                response_list.append(response)
            ground_truth_list.append(ground_truth)
        try:
            result = metrics(response_list, ground_truth_list)
        except Exception as e:
            raise RuntimeError('metrics fails!')
        else:
            self._write_output(result)

    def _parse_manifest(self):
        with open(self.manifest, 'r') as f:
            manifest = json.load(f)
            for i, val in enumerate(manifest['spec']['inputs']):
                if val['name'] in self.input_list:
                    continue
                self.input_list.append(val['name'])
                self.input_size_list.append(val['dimValue'])
                self.dtype_list.append(val['dataType'])

    def _post_request(self, payload):
        response = requests.post(self.server, payload)
        if response.status_code == 200:
            result_json = json.loads(response.text)
            result = result_json[
                'predictions'] if 'predictions' in result_json else result_json[
                    'outputs']
            return result
        else:
            print("serving error:", response.text)
            return ''

    def _write_output(self, result):
        res = json.dumps(result)
        out = {
            "name": self.name,
            "result": res,
        }
        filename = ("%s.json" % self.name)
        file = os.path.join(self.output, filename)
        with open(file, 'w') as f:
            json.dump(out, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process args for evaluation')
    parser.add_argument("--output", "-o", help="set output directory")
    parser.add_argument("--data", "-d", help="set dataset directory")
    parser.add_argument("--name", "-n", help="set function name")
    parser.add_argument("--server", "-s", help="set serving server address")
    parser.add_argument("--manifest", "-m", help="set the manifest path of model")
    args = parser.parse_args()

    server = "http://{s}/predict".format(s=args.server)

    evaluation = Evaluation(args.name, args.data, server, args.output, args.manifest)

    evaluation.execute()
