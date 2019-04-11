
from keras.applications.vgg16 import VGG16
from keras.models import Model,Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.recurrent import GRU
from keras.layers.core import Reshape,Dense,Flatten,Permute,Lambda,Activation
from keras.layers.wrappers import Bidirectional,TimeDistributed
from keras.layers import Input
from keras.optimizers import Adam,SGD
from keras import backend as K
from keras import regularizers 
import tensorflow as tf 
from keras.callbacks import EarlyStopping,ModelCheckpoint,Callback

def rpn_loss_regr(y_true,y_pred):
    """
    smooth L1 loss
  
    y_ture [1][HXWX9][3] (class,regr)
    y_pred [1][HXWX9][2] (reger)
    """   
    
    sigma=9.0
    
    cls = y_true[0,:,0]
    regr = y_true[0,:,1:3]
    regr_keep = tf.where(K.equal(cls,1))[:,0]
    regr_true = tf.gather(regr,regr_keep)
    regr_pred = tf.gather(y_pred[0],regr_keep)
    diff = tf.abs(regr_true-regr_pred)
    #tf.less逐元素返回是否x<y
    less_one = tf.cast(tf.less(diff,1.0/sigma),'float32')
    ### 这是个什么loss? 字面上的意思是防止loss过大，将loss>1的先减掉一个常数
    loss = less_one * 0.5 * diff**2 * sigma   + tf.abs(1-less_one) * (diff -0.5/sigma)
    loss = K.sum(loss,axis=1)
    
    return K.switch(tf.size(loss)>0,K.mean(loss),K.constant(0.0))

def rpn_loss_cls(y_true,y_pred):
    """
    softmax loss
    
    y_true [1][1][HXWX9] class
    y_pred [1][HXWX9][2] class 
    
    9 means 有9个anchor
    """ 
    y_true = y_true[0][0]
    cls_keep = tf.where(tf.not_equal(y_true,-1))[:,0]
    cls_true = tf.gather(y_true,cls_keep)
    cls_pred = tf.gather(y_pred[0],cls_keep)
    cls_true = tf.cast(cls_true,'int64')
    #loss = K.sparse_categorical_crossentropy(cls_true,cls_pred,from_logits=True)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = cls_true,logits=cls_pred)
    ## 如果loss的维度不为None
    return K.switch(tf.size(loss)>0,K.clip(K.mean(loss),0,10),K.constant(0.0))

def rpn_loss_name(y_true,y_pred):
    """
    softmax loss
    
    y_true [1][1][HXWX9] class
    y_pred [1][HXWX9][16] class 
    """     
    y_true = y_true[0][0]
    cls_keep = tf.where(tf.not_equal(y_true,-1))[:,0]
    cls_true = tf.gather(y_true,cls_keep)
    cls_pred = tf.gather(y_pred[0],cls_keep)
    cls_true = tf.cast(cls_true,'int64')
    #loss = K.sparse_categorical_crossentropy(cls_true,cls_pred,from_logits=True)
    cls_true_one_hot = tf.one_hot(cls_true,depth=16,on_value=1.0,name='name_index_one_hot')
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = cls_true_one_hot,logits=cls_pred)
#     loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = cls_true,logits=cls_pred)

    return K.switch(tf.size(loss)>0,K.clip(K.mean(loss),0,10),K.constant(0.0))


def nn_base(input,trainable):
    base_model = VGG16(weights='imagenet',include_top=False,input_shape = input)
#     base_model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    if(trainable ==False):
        for ly in base_model.layers:
            ly.trainable = False
    # 当我们重用pretrain model 可以这样给出input output
    return base_model.input,base_model.get_layer('block5_conv3').output

def reshape(x):
    b = tf.shape(x)
    x = tf.reshape(x,[b[0]*b[1],b[2],b[3]])
    return x

def reshape2(x):
    x1,x2 = x
    b = tf.shape(x2)
    x = tf.reshape(x1,[b[0],b[1],b[2],256])
    return x 

def reshape3(x):
    b = tf.shape(x)
    x = tf.reshape(x,[b[0],b[1]*b[2]*10,2])
    return x 

def reshape4(x):
    b = tf.shape(x)
    x = tf.reshape(x,[b[0],b[1]*b[2]*10,16])
    return x

def rpn(base_layers):
    
    x = Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',
               name='rpn_conv1')(base_layers)
    
    x1 = Lambda(reshape,output_shape=(None,512))(x) 
    
    x2 = Bidirectional(GRU(128,return_sequences=True),name='blstm')(x1)

    x3 = Lambda(reshape2,output_shape=(None,None,256))([x2,x])
    x3 = Conv2D(512,(1,1),padding='same',activation='relu',name='lstm_fc')(x3)

    ## 这里相当于对 feature map 每一个pixels 求它的 pred
    cls = Conv2D(10*2,(1,1),padding='same',activation='linear',name='rpn_class')(x3)
    regr = Conv2D(10*2,(1,1),padding='same',activation='linear',name='rpn_regress')(x3)
    

    cls = Lambda(reshape3,output_shape=(None,2),name='rpn_class_reshape')(cls)
    cls_prod = Activation('softmax',name='rpn_cls_softmax')(cls)

   
    ## lambda output_shape是忽略batch 的维度的
    regr = Lambda(reshape3,output_shape=(None,2),name='rpn_regress_reshape')(regr)
    
    return cls,regr,cls_prod



def rpn_with_name(base_layers):
    
    x = Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',
               name='rpn_conv1')(base_layers)
    
    x1 = Lambda(reshape,output_shape=(None,512))(x) 
    
    x2 = Bidirectional(GRU(128,return_sequences=True),name='blstm')(x1)

    x3 = Lambda(reshape2,output_shape=(None,None,256))([x2,x])
    x3 = Conv2D(512,(1,1),padding='same',activation='relu',name='lstm_fc')(x3)

    cls = Conv2D(10*2,(1,1),padding='same',activation='linear',name='rpn_class')(x3)
    regr = Conv2D(10*2,(1,1),padding='same',activation='linear',name='rpn_regress')(x3)
    name_cls =  Conv2D(10*16,(1,1),padding='same',activation='linear',name='rpn_name_class')(x3)

    cls = Lambda(reshape3,output_shape=(None,2),name='rpn_class_reshape')(cls)
    cls_prod = Activation('softmax',name='rpn_cls_softmax')(cls)

    regr = Lambda(reshape3,output_shape=(None,2),name='rpn_regress_reshape')(regr)
    name_cls  = Lambda(reshape4,output_shape=(None,16),name='rpn_name_cls_reshape')(name_cls)
    name_cls_prod = Activation('softmax',name='rpn_name_cls_softmax')(name_cls)

    return cls,regr,cls_prod,name_cls,name_cls_prod


def get_model():
    inp,nn = nn_base((None,None,3),trainable=True)
    cls,regr,cls_prod = rpn(nn)
    basemodel =  Model(inp,[cls,regr,cls_prod])
    return basemodel



def get_model_with_labels():
    inp,nn = nn_base((None,None,3),trainable=False)
    cls,regr,cls_prod,name_cls,name_cls_prod = rpn_with_name(nn)
    basemodel = Model(inp,[cls,regr,cls_prod,name_cls,name_cls_prod])
    return basemodel
# utils.get_session(gpu_fraction=0.8)