import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D,Bidirectional,LSTM,Reshape,BatchNormalization,Flatten,Dropout,Dense
from keras.layers import add
from keras.utils import plot_model

from keras.models import Model
import copy
import warnings
warnings.filterwarnings('ignore')
import cv2
import keras
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten, Input
from keras.layers import Conv2D, Activation, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input, ResNet50
import keras
from keras import optimizers
from keras import backend


def resnet_model(size = (256,2048,1)):
    ''' This model is build using keras module from the paper https://arxiv.org/pdf/1910.12590.pdf
    inputs are to be resized of 256,2048,1  and the no of classification items. I have fixed to binary as default
    output is the model
    '''
    input  = Input(shape = size)
    bnEps=2e-5
    bnMom=0.9


    c1 = Conv2D(64, (7,7), padding='same',strides=2,activation='relu', use_bias=False,kernel_initializer='glorot_uniform')(input)
    b1 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c1)

    c2 = conv1 = Conv2D(32, (3,3),strides=2, padding='same', use_bias=False,kernel_initializer='glorot_uniform')(input)
    b2 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c2)
    a2 = Activation('relu')(b2)

    c3 = conv1 = Conv2D(64, (3,3), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a2)
    b3 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c3)
    a3 = Activation('relu')(b3)

    c4 = conv1 = Conv2D(64, (3,3), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a3)
    b4 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c4)

    m1  = add([c1, b4])
    a4 = Activation('relu')(m1)


    #-----------------------------------------------layer 2----------------------------------------------------------------------------

    c1 = Conv2D(128, (3,3),strides=2, padding='same',activation='relu', use_bias=False,kernel_initializer='glorot_uniform')(a4)
    b1 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c1)

    c2 = conv1 = Conv2D(64, (3,3),strides=2, padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a4)
    b2 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c2)
    a2 = Activation('relu')(b2)

    c3 = conv1 = Conv2D(128, (3,3), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a2)
    b3 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c3)
    a3 = Activation('relu')(b3)

    c4 = conv1 = Conv2D(128, (3,3), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a3)
    b4 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c4)

    m1  = add([c1, b4])
    a4 = Activation('relu')(m1)

    #----------------------------------------------layer 3------------------------------------------------------------------------------

    c1 = Conv2D(128, (3,3),strides = (1,2) ,padding='same',activation='relu', use_bias=False,kernel_initializer='glorot_uniform')(a4)
    b1 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c1)

    c2 = conv1 = Conv2D(128, (3,3),strides = (1,2), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a4)
    b2 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c2)
    a2 = Activation('relu')(b2)

    c3 = conv1 = Conv2D(128, (3,3), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a2)
    b3 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c3)
    a3 = Activation('relu')(b3)


    c4 = conv1 = Conv2D(128, (3,3), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a3)
    b4 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c4)

    m1  = add([c1, b4])
    a4 = Activation('relu')(m1)

    #-------------------------------------------layer 4---------------------------------------------------------------------------------

    c1 = Conv2D(64, (3,3),strides = (2,2) ,padding='same',activation='relu', use_bias=False,kernel_initializer='glorot_uniform')(a4)
    b1 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c1)

    c2 = conv1 = Conv2D(128, (3,3),strides = (2,2), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a4)
    b2 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c2)
    a2 = Activation('relu')(b2)

    c3 = conv1 = Conv2D(64, (3,3), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a2)
    b3 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c3)
    a3 = Activation('relu')(b3)


    c4 = conv1 = Conv2D(64, (3,3), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a3)
    b4 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c4)

    m1  = add([c1, b4])
    a4 = Activation('relu')(m1)

    #-------------------------------------------layer 5-----------------------------------------------------------------------------------
    c1 = Conv2D(32, (3,3),strides = (2,2) ,padding='same',activation='relu', use_bias=False,kernel_initializer='glorot_uniform')(a4)
    b1 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c1)

    c2 = conv1 = Conv2D(64, (3,3),strides = (2,2), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a4)
    b2 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c2)
    a2 = Activation('relu')(b2)

    c3 = conv1 = Conv2D(64, (3,3), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a2)
    b3 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c3)
    a3 = Activation('relu')(b3)


    c4 = conv1 = Conv2D(32, (3,3), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a3)
    b4 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c4)

    m1  = add([c1, b4])
    a4 = Activation('relu')(m1)

    #-----------------------------------------layer 6-------------------------------------------------------------------------
    c1 = Conv2D(16, (3,3),strides = (2,2) ,padding='same',activation='relu', use_bias=False,kernel_initializer='glorot_uniform')(a4)
    b1 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c1)

    c2 = conv1 = Conv2D(32, (3,3),strides = (2,2), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a4)
    b2 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c2)
    a2 = Activation('relu')(b2)

    c3 = conv1 = Conv2D(32, (3,3), padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a2)
    b3 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c3)
    a3 = Activation('relu')(b3)


    c4 = conv1 = Conv2D(16, (3,3), padding='same',use_bias=False,kernel_initializer='glorot_uniform')(a3)
    b4 = BatchNormalization(epsilon=bnEps, momentum=bnMom)(c4)

    m1  = add([c1, b4])
    a4 = Activation('relu')(m1)

    f = Flatten()(a4)
    # f = Reshape((int(8192/2), 1))(f)

    # #-----------------------------------------layer7---------------------------------------------------------------------------
    # bi1 = Bidirectional(CuDNNLSTM(512, return_sequences=True))(f)
    bi1 = Dense(1024,activation='relu')(f)
    d1  = Dropout(0.2)(bi1)

    # bi2 = Bidirectional(CuDNNLSTM(512))(d1)
    bi2 = Dense(512,activation='relu')(d1)
    d2 = Dropout(0.4)(bi2)

    out = Dense(2,activation='sigmoid')(d2)

    # create model
    model = Model(inputs=input, outputs=out)
    return model


# calculate fbeta score for multi-class/label classification
def fbeta(y_true, y_pred, beta=2):
	# clip predictions
	y_pred = backend.clip(y_pred, 0, 1)
	# calculate elements
	tp = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)), axis=1)
	fp = backend.sum(backend.round(backend.clip(y_pred - y_true, 0, 1)), axis=1)
	fn = backend.sum(backend.round(backend.clip(y_true - y_pred, 0, 1)), axis=1)
	# calculate precision
	p = tp / (tp + fp + backend.epsilon())
	# calculate recall
	r = tp / (tp + fn + backend.epsilon())
	# calculate fbeta, averaged across each class
	bb = beta ** 2
	fbeta_score = backend.mean((1 + bb) * (p * r) / (bb * p + r + backend.epsilon()))
	return fbeta_score
