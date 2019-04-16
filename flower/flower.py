import os
import json
import argparse
import requests
import tensorflow as tf

def input_fn(tfrecords_path,batch_size=1):
    dataset = tf.data.TFRecordDataset(tfrecords_path)
    dataset = dataset.map(parser)
#     dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    return iterator

def parser(record):
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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def execute_evaluation(evaluation):
    # add user code here:
    # 1. data = load_data(evaluation.data)
    # 2. payload = {"instances":[{"input": data.astype(np.float32).tolist()}]}
    # 3. response = evaluation.post_request(payload)
    # 4. result = metrics(response)
    # 5. evaluation.write_output(result)
    iterator = input_fn(FLAGS.tfrecords_path)
    x,label = iterator.get_next()
    logits = tf.placeholder(tf.float32,[None,5])
    acc, acc_op = tf.metrics.accuracy(labels=label, predictions=tf.argmax(logits,axis=1))
    greater = tf.cast(tf.math.greater(logits,0.5),tf.int64)
    precision,pre_op = tf.metrics.precision_at_k(tf.cast(tf.expand_dims(tf.one_hot(label,depth=5),axis=0),tf.int64), k=4,predictions=greater)
    ## 设置 init 和 saver,
    init = [tf.global_variables_initializer(),tf.local_variables_initializer()]
    with tf.Session() as sess:
        sess.run(init)
        # edited
        temp_x,temp_label = sess.run([x,label])
        p_data0 = {'inputs_1':temp_x}
        param = {"instances":[p_data0]}
        predict_request = json.dumps(param,cls=NumpyEncoder)
        prediction = evaluation.post_request(predict_request)

        temp_logits = np.array([prediction])
        temp_acc,_ = sess.run([acc,acc_op],feed_dict={label:temp_label,logits:temp_logits})
        print('accuracy: %f,precision: %f'%(temp_acc,temp_pre))


        # try:
        #     while True:
        #         temp_x,temp_label = sess.run([x,label])
        #         p_data0 = {'inputs_1':temp_x}
        #         param = {"instances":[p_data0]}
        #         predict_request = json.dumps(param,cls=NumpyEncoder)
        #         response = evaluation.post_request(predict_request)

        #         # response = requests.post(FLAGS.SERVER_URL, data=predict_request)
        #         # response.raise_for_status()
        #         prediction = response.json()['predictions'][0]
        #         temp_logits = np.array([prediction])
        #         temp_acc,_ = sess.run([acc,acc_op],
        #                               feed_dict={label:temp_label,
        #                                         logits:temp_logits})
        # except tf.errors.OutOfRangeError as info:
        #     print('accuracy: %f,precision: %f'%(temp_acc,temp_pre))


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
