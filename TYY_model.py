import logging
import sys
import numpy as np
from keras.models import Model
from keras.layers import Input, Activation, add, Dense, Flatten, Dropout, Multiply, Embedding, Lambda,Add, Concatenate
from keras.layers.convolutional import Conv2D, AveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.applications.mobilenet import MobileNet
from densenet import *


sys.setrecursionlimit(2 ** 20)
np.random.seed(2 ** 10)


class TYY_MobileNet_reg:
    def __init__(self, image_size, alpha):
        
        self._dropout_probability = 0
        self._weight_decay = 0.0005
        self._use_bias = False
        self._weight_init = "he_normal"
        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)
        self.alpha = alpha

#    def create_model(self):
    def __call__(self):
        logging.debug("Creating model...")

        inputs = Input(shape=self._input_shape)
        model_mobilenet = MobileNet(input_shape=self._input_shape, alpha=self.alpha, depth_multiplier=1, dropout=1e-3, include_top=False, weights=None, input_tensor=None, pooling=None)
        x = model_mobilenet(inputs)
        #flatten = Flatten()(x)
        
        feat_a = Conv2D(20,(1,1),activation='relu')(x)
        feat_a = Flatten()(feat_a)
        feat_a = Dropout(0.2)(feat_a)
        feat_a = Dense(32,activation='relu')(feat_a)

        pred_a = Dense(1,name='pred_a')(feat_a)
        model = Model(inputs=inputs, outputs=[pred_a])


        return model

class TYY_MobileNet_dex:
    def __init__(self, image_size, alpha, num_neu):
        
        self._dropout_probability = 0
        self._weight_decay = 0.0005
        self._use_bias = False
        self._weight_init = "he_normal"
        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)
        self.alpha = alpha
        self.num_neu = num_neu

#    def create_model(self):
    def __call__(self):
        logging.debug("Creating model...")

        inputs = Input(shape=self._input_shape)
        model_mobilenet = MobileNet(input_shape=self._input_shape, alpha=self.alpha, depth_multiplier=1, dropout=1e-3, include_top=False, weights=None, input_tensor=None, pooling=None)
        x = model_mobilenet(inputs)
        #flatten = Flatten()(x)
        
        feat_a = Conv2D(20,(1,1),activation='relu')(x)
        feat_a = Flatten()(feat_a)
        feat_a = Dropout(0.2)(feat_a)
        feat_a = Dense(32,activation='relu')(feat_a)

        pred_a_softmax = Dense(self.num_neu,activation='softmax', name='pred_a_softmax')(feat_a)

        def mult_fixed_range(x, num_neu):
            a = x[:,0]*0
            for i in range(0,num_neu):
                a = a+i*x[:,i]*101/num_neu
            return a
        pred_a = Lambda(mult_fixed_range, arguments={'num_neu': self.num_neu}, output_shape=(1,), name='pred_a')(pred_a_softmax)
        
        model = Model(inputs=inputs, outputs=[pred_a_softmax,pred_a])

        return model

class TYY_DenseNet_reg:
    def __init__(self, image_size, depth):
        
        self._dropout_probability = 0
        self._weight_decay = 0.0005
        self._use_bias = False
        self._weight_init = "he_normal"
        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)
        self.depth = depth

#    def create_model(self):
    def __call__(self):
        logging.debug("Creating model...")

        inputs = Input(shape=self._input_shape)
        model_densenet = DenseNet(input_shape=self._input_shape, depth=self.depth, include_top=False, weights=None, input_tensor=None)
        flatten = model_densenet(inputs)
        
        feat_a = Dense(128,activation='relu')(flatten)
        feat_a = Dropout(0.2)(feat_a)
        feat_a = Dense(32,activation='relu',name='feat_a')(feat_a)

        pred_a = Dense(1,name='pred_a')(feat_a)
        model = Model(inputs=inputs, outputs=[pred_a])
        
        return model

class TYY_DenseNet_dex:
    def __init__(self, image_size, depth, num_neu):
        
        self._dropout_probability = 0
        self._weight_decay = 0.0005
        self._use_bias = False
        self._weight_init = "he_normal"
        if K.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)
        self.depth = depth
        self.num_neu = num_neu

#    def create_model(self):
    def __call__(self):
        logging.debug("Creating model...")

        inputs = Input(shape=self._input_shape)
        model_densenet = DenseNet(input_shape=self._input_shape, depth=self.depth, include_top=False, weights=None, input_tensor=None)
        flatten = model_densenet(inputs)
        
        feat_a = Dense(128,activation='relu')(flatten)
        feat_a = Dropout(0.2)(feat_a)
        feat_a = Dense(32,activation='relu',name='feat_a')(feat_a)

        pred_a_softmax = Dense(self.num_neu,activation='softmax', name='pred_a_softmax')(feat_a)

        def mult_fixed_range(x, num_neu):
            a = x[:,0]*0
            for i in range(0,num_neu):
                a = a+i*x[:,i]*101/num_neu
            return a
        pred_a = Lambda(mult_fixed_range, arguments={'num_neu': self.num_neu}, output_shape=(1,), name='pred_a')(pred_a_softmax)
        
        model = Model(inputs=inputs, outputs=[pred_a_softmax,pred_a])
        
        return model

