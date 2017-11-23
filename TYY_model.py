# This code is imported from the following project: https://github.com/asmith26/wide_resnets_keras

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
sys.setrecursionlimit(2 ** 20)
np.random.seed(2 ** 10)


class TYY_2stream:
    def __init__(self, image_size):
        
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


#    def create_model(self):
    def __call__(self):
        logging.debug("Creating model...")


        inputs = Input(shape=self._input_shape)

        x = Conv2D(32,(3,3),activation='relu')(inputs)
        x = MaxPooling2D(2,2)(x)
        x = Conv2D(32,(3,3),activation='relu')(x)
        x = MaxPooling2D(2,2)(x)
        x = Conv2D(64,(3,3),activation='relu')(x)
        x = MaxPooling2D(2,2)(x)
        x = Conv2D(64,(3,3),activation='relu')(x)
        

        y = Conv2D(32,(3,3),activation='relu')(inputs)
        y = MaxPooling2D(2,2)(y)
        y = Conv2D(32,(3,3),activation='relu')(y)
        y = MaxPooling2D(2,2)(y)
        y = Conv2D(64,(3,3),activation='relu')(y)
        y = MaxPooling2D(2,2)(y)
        y = Conv2D(64,(3,3),activation='tanh')(y)
        
        z = Multiply()([x,y])
        z = BatchNormalization(axis=self._channel_axis)(z)

        # Classifier block
        flatten = Flatten()(z)
        feat_g = Dropout(0.2)(flatten)
        feat_g = Dense(32)(feat_g)
        feat_a = Dropout(0.2)(flatten)
        feat_a = Dense(32)(feat_a)

        predictions_g = Dense(units=2, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax",name='gender')(feat_g)
        predictions_a = Dense(units=21, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax",name='age')(feat_a)

        model = Model(inputs=inputs, outputs=[predictions_g, predictions_a])



        return model


class TYY_2stream_centerloss:
    def __init__(self, image_size):
        
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


#    def create_model(self):
    def __call__(self):
        logging.debug("Creating model...")


        inputs = Input(shape=self._input_shape)

        x = Conv2D(32,(3,3),activation='relu')(inputs)
        x = MaxPooling2D(2,2)(x)
        x = Conv2D(32,(3,3),activation='relu')(x)
        x = MaxPooling2D(2,2)(x)
        x = Conv2D(64,(3,3),activation='relu')(x)
        x = MaxPooling2D(2,2)(x)
        x = Conv2D(64,(3,3),activation='relu')(x)
        

        y = Conv2D(32,(3,3),activation='relu')(inputs)
        y = MaxPooling2D(2,2)(y)
        y = Conv2D(32,(3,3),activation='relu')(y)
        y = MaxPooling2D(2,2)(y)
        y = Conv2D(64,(3,3),activation='relu')(y)
        y = MaxPooling2D(2,2)(y)
        y = Conv2D(64,(3,3),activation='tanh')(y)
        
        z = Multiply()([x,y])
        z = BatchNormalization(axis=self._channel_axis)(z)

        # Classifier block
        flatten = Flatten()(z)
        feat_g = Dropout(0.2)(flatten)
        feat_g = Dense(32)(feat_g)
        feat_a = Dropout(0.2)(flatten)
        feat_a = Dense(32)(feat_a)

        predictions_g = Dense(units=2, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax",name='gender')(feat_g)
        predictions_a = Dense(units=21, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax",name='age')(feat_a)

        model = Model(inputs=inputs, outputs=[predictions_g, predictions_a])

        
        input_gender_target = Input(shape=(1,))
        centers = Embedding(2,32)(input_gender_target)
        l2_loss = Lambda(lambda x: K.sum(K.square(x[0]-x[1][:,0]),1,keepdims=True),name='l2_loss')([feat_g,centers])
        model_centorloss = Model(inputs=[inputs,input_gender_target],outputs=[predictions_g, predictions_a,l2_loss])
        
        return model,model_centorloss


class TYY_2stream_lineloss:
    def __init__(self, image_size):
        
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


#    def create_model(self):
    def __call__(self):
        logging.debug("Creating model...")


        inputs = Input(shape=self._input_shape)

        x = Conv2D(32,(3,3),activation='relu')(inputs)
        x = MaxPooling2D(2,2)(x)
        x = Conv2D(32,(3,3),activation='relu')(x)
        x = MaxPooling2D(2,2)(x)
        x = Conv2D(64,(3,3),activation='relu')(x)
        x = MaxPooling2D(2,2)(x)
        x = Conv2D(64,(3,3),activation='relu')(x)
        

        y = Conv2D(32,(3,3),activation='relu')(inputs)
        y = MaxPooling2D(2,2)(y)
        y = Conv2D(32,(3,3),activation='relu')(y)
        y = MaxPooling2D(2,2)(y)
        y = Conv2D(64,(3,3),activation='relu')(y)
        y = MaxPooling2D(2,2)(y)
        y = Conv2D(64,(3,3),activation='tanh')(y)
        
        z = Multiply()([x,y])
        z = BatchNormalization(axis=self._channel_axis)(z)

        # Classifier block
        flatten = Flatten()(z)
        feat_g = Dropout(0.2)(flatten)
        feat_g = Dense(32)(feat_g)
        feat_a = Dropout(0.2)(flatten)
        feat_a = Dense(32)(feat_a)

        predictions_g = Dense(units=2, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax",name='gender')(feat_g)
        predictions_a = Dense(units=21, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax",name='age')(feat_a)

        model = Model(inputs=inputs, outputs=[predictions_g, predictions_a])

        

        lambda_softmax_g = 1
        lambda_softmax_a = 1
        lambda_p = 1
        lambda_line = 0.2
        lambda_C1 = 0.2
        lambda_C1bar = 0.
        input_target = Input(shape=(1,))

        D = feat_a
        D = Dense(128)(D)
        C1 = Embedding(21,128)(input_target)
        C1bar = Lambda(lambda x:x[:,0])(C1)
        #C1toC1bar_layer = Dense(2,activation='tanh',weights=[np.eye(2),np.asarray([1,1])])
        #C1bar = C1toC2_layer(C1bar)
        C1bar = Dense(128)(C1bar)
        #C1bar = Dense(32)(C1bar)
                

        V1 = Lambda(lambda x: x[0][:,0]-x[1])([C1,C1bar])
        V2 = Lambda(lambda x: x[0][:,0]-x[1])([C1,D])
        V3 = Lambda(lambda x: x[0]-x[1])([C1bar,D])
        V1_sqvalue = Lambda(lambda x: K.sum(K.square(x),1,keepdims=True))(V1)
        V2_sqvalue = Lambda(lambda x: K.sum(K.square(x),1,keepdims=True))(V2)
        V3_sqvalue = Lambda(lambda x: K.sum(K.square(x),1,keepdims=True))(V3)

        V1V2_sqvalue = Lambda(lambda x:K.square(K.sum(x[0]*x[1],1,keepdims=True)))([V1,V2])
        U_sqvalue = Lambda(lambda x: x[0]/(0.001+x[1]))([V1V2_sqvalue,V1_sqvalue])
        P_sqvalue = Lambda(lambda x: x[0]-x[1])([V2_sqvalue,U_sqvalue])
        p_loss = Lambda(lambda x:lambda_line*x[0]+lambda_C1*x[1]+lambda_C1bar*x[2],name='p_loss')([P_sqvalue,V2_sqvalue,V3_sqvalue])

        model_lineloss = Model(inputs=[inputs,input_target],outputs=[predictions_g, predictions_a,p_loss])
        model_lineloss.compile(optimizer=SGD(lr=0.05), loss=["categorical_crossentropy","categorical_crossentropy", lambda y_true,y_pred: y_pred],loss_weights=[lambda_softmax_g,lambda_softmax_a,lambda_p],metrics=['accuracy'])
        #model_lineloss.compile(optimizer=Adam(), loss=["categorical_crossentropy","categorical_crossentropy", lambda y_true,y_pred: y_pred],loss_weights=[lambda_softmax_g,lambda_softmax_a,lambda_p],metrics=['accuracy'])

        return model,model_lineloss



class TYY_1stream:
    def __init__(self, image_size):
        
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


#    def create_model(self):
    def __call__(self):
        logging.debug("Creating model...")


        inputs = Input(shape=self._input_shape)

        x = Conv2D(32,(3,3),activation='relu')(inputs)
        x = MaxPooling2D(2,2)(x)
        x = Conv2D(32,(3,3),activation='relu')(x)
        x = MaxPooling2D(2,2)(x)
        x = Conv2D(64,(3,3),activation='relu')(x)
        x = MaxPooling2D(2,2)(x)
        x = Conv2D(64,(3,3),activation='relu')(x)
        x = BatchNormalization(axis=self._channel_axis)(x)

        # Classifier block
        pool = AveragePooling2D(pool_size=(4, 4), strides=(1, 1), padding="same")(x)
        flatten = Flatten()(pool)
        predictions_g = Dense(units=2, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax")(flatten)
        predictions_a = Dense(units=21, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax")(flatten)

        model = Model(inputs=inputs, outputs=[predictions_g, predictions_a])



        return model

class TYY_MobileNet:
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
        
        feat_g = Conv2D(20,(1,1),activation='relu')(x)
        feat_g = Flatten()(feat_g)
        feat_g = Dropout(0.2)(feat_g)
        feat_g = Dense(32,activation='relu')(feat_g)
        
        feat_a = Conv2D(20,(1,1),activation='relu')(x)
        feat_a = Flatten()(feat_a)
        feat_a = Dropout(0.2)(feat_a)
        feat_a = Dense(32,activation='relu')(feat_a)

        predictions_g = Dense(units=2, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax")(feat_g)
        predictions_a = Dense(units=21, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax")(feat_a)

        model = Model(inputs=inputs, outputs=[predictions_g, predictions_a])


        return model

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
        
        feat_g = Conv2D(20,(1,1),activation='relu')(x)
        feat_g = Flatten()(feat_g)
        feat_g = Dropout(0.2)(feat_g)
        feat_g = Dense(32,activation='relu')(feat_g)
        
        feat_a = Conv2D(20,(1,1),activation='relu')(x)
        feat_a = Flatten()(feat_a)
        feat_a = Dropout(0.2)(feat_a)
        feat_a = Dense(32,activation='relu')(feat_a)

        predictions_g = Dense(units=2, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax")(feat_g)
        predictions_a = Dense(units=1, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay))(feat_a)

        model = Model(inputs=inputs, outputs=[predictions_g, predictions_a])


        return model


class TYY_MobileNet_2stream:
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
        model_mobilenet1 = MobileNet(input_shape=self._input_shape, alpha=self.alpha, depth_multiplier=1, dropout=1e-3, include_top=False, weights=None, input_tensor=None, pooling='max')
        model_mobilenet1.name = 'model_mobilenet1'
        model_mobilenet2 = MobileNet(input_shape=self._input_shape, alpha=self.alpha, depth_multiplier=1, dropout=1e-3, include_top=False, weights=None, input_tensor=None, pooling='max')
        model_mobilenet2.name = 'model_mobilenet2'

        feat_g = model_mobilenet1(inputs)
        feat_a = model_mobilenet2(inputs)
        #flatten = Flatten()(x)
        predictions_g = Dense(units=2, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax")(feat_g)
        predictions_a = Dense(units=21, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax")(feat_a)

        model = Model(inputs=inputs, outputs=[predictions_g, predictions_a])


        return model

class SPY_net:
    def __init__(self, image_size,stage_num,feat_dim):
        
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


        self.stage_num = stage_num
        self.feat_dim = feat_dim

#    def create_model(self):
    def __call__(self):
        logging.debug("Creating model...")


        inputs = Input(shape=self._input_shape)

        
        x = Conv2D(32,(3,3),activation='relu')(inputs)
        x_layer1 = MaxPooling2D(2,2)(x)
        x = Conv2D(32,(3,3),activation='relu')(x_layer1)
        x_layer2 = MaxPooling2D(2,2)(x)
        x = Conv2D(32,(3,3),activation='relu')(x_layer2)
        x_layer3 = MaxPooling2D(2,2)(x)
        x = Conv2D(32,(3,3),activation='relu')(x_layer3)
        x = BatchNormalization(axis=self._channel_axis)(x)

        # Classifier block
        x_layer4 = Flatten()(x)
        
        pred_a_s1 = Dense(units=self.stage_num[0], kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax",name='age_stage1')(x_layer4)

        
        x_layer2_mix = Conv2D(1,(1,1),activation='relu')(x_layer2)
        x_layer2_mix = Flatten()(x_layer2_mix)
        x_layer2_mix = BatchNormalization(axis=self._channel_axis)(x_layer2_mix)
        x_layer2_mix = Dense(self.feat_dim,activation='relu')(x_layer2_mix)
        x_layer4_mix = Dense(self.feat_dim,activation='relu')(x_layer4)
        x_layer5 = Concatenate()([x_layer2_mix,x_layer4_mix])

        pred_a_s2 = Dense(units=self.stage_num[1], kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax",name='age_stage2')(x_layer5)

        x_layer1_mix = Conv2D(1,(1,1),activation='relu')(x_layer1)
        x_layer1_mix = Flatten()(x_layer1_mix)
        x_layer1_mix = BatchNormalization(axis=self._channel_axis)(x_layer1_mix)
        x_layer1_mix = Dense(self.feat_dim,activation='relu')(x_layer1_mix)
        x_layer5_mix = Dense(self.feat_dim,activation='relu')(x_layer5)
        x_layer6 = Concatenate()([x_layer1_mix,x_layer5_mix])

        pred_a_s3 = Dense(units=self.stage_num[2], kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax",name='age_stage3')(x_layer6)
        
        

        pred_g = Dense(units=2, kernel_initializer=self._weight_init, use_bias=self._use_bias,
                              kernel_regularizer=l2(self._weight_decay), activation="softmax",name='gender')(x_layer6)
        

        model = Model(inputs=inputs, outputs=[pred_g, pred_a_s1, pred_a_s2, pred_a_s3])

        #model.summary()
        #sys.exit()
        return model