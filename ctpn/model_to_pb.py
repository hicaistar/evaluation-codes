from keras.models import load_model
import tensorflow as tf
from tensorflow.python.platform import gfile
import os 
from keras import backend as K
import numpy as np



tf.app.flags.DEFINE_string('input_path',  '../model/ctpnlstm-20.hdf5','')
tf.app.flags.DEFINE_string('output_dir',  '../model/','')

FLAGS = tf.app.flags.FLAGS


#转换函数
def h5_to_pb(input_path,output_dir,out_prefix = "output_",log_tensorboard = True):
    h5_model = load_model(input_path)
    model_name = os.path.basename(input_path)
    if os.path.exists(output_dir) == False:
        os.mkdir(output_dir)
    out_nodes = []
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i],out_prefix + str(i + 1))
    sess = K.get_session()
    from tensorflow.python.framework import graph_util,graph_io
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess,init_graph,out_nodes)
    graph_io.write_graph(main_graph,output_dir,name = model_name,as_text = False)
    if log_tensorboard:
        from tensorflow.python.tools import import_pb_to_tensorboard
        import_pb_to_tensorboard.import_to_tensorboard(os.path.join(output_dir,model_name),output_dir)

def load_pb(pb_file_path):
    sess = tf.Session()
    with gfile.FastGFile(pb_file_path, 'rb') as f:
        try:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        except Exception as err:
            print(err)
    return sess

def main(argv=None):
    h5_to_pb(FLAGS.input_path,FLAGS.output_dir)

if __name__=='__main__':
    tf.app.run()
