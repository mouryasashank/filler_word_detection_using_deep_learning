import numpy as np
from keras.layers import MaxPooling2D,Bidirectional,LSTM,Reshape,BatchNormalization,Flatten,Dropout,Dense,Input,Conv2D, Activation, GlobalAveragePooling2D
from keras.layers import add
from keras.utils import plot_model
import copy
import warnings
warnings.filterwarnings('ignore')
import cv2
import keras
from keras import backend as K
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input, ResNet50
import keras
from keras import optimizers
from keras import backend
import keras
from keras import optimizers
from keras.layers import GaussianNoise
from keras.regularizers import l2,l1
import os
import numpy as np
import librosa
from pydub import AudioSegment
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
import os
import matplotlib
import pylab
import librosa
import librosa.display
import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks
import cv2
import os
import numpy as np
import matplotlib 
from glob import glob
from tqdm import tqdm
import parselmouth
import time
from tqdm import tqdm
from glob import glob
import pandas as pd
import os
import numpy as np
import librosa
from pydub import AudioSegment
from tqdm import tqdm
from glob import glob
import plotly.graph_objects as go
from shutil import copyfile


def praat_script(i):
  clean_complete_dir('/content/rough')
  # all_files = glob('/content/chunked_audio_files/*.wav')
  # for i in tqdm(all_files):
  name = i.split('/')[-1]
  dst = 'rough/'+name
  copyfile(i, dst)
  path = '/content/rough'

  try:
        objects, output = parselmouth.praat.run_file('/content/nucleus.praat',-25,2,0.3,'yes',path, capture_output=True)
        outputs = output.split('\n')[1:]
        outputs = outputs[0].split(',')
        return float(outputs[4])

  except:
        print('error occured at',i)
        return False
       
  clean_complete_dir('/content/rough')

def make_prediction(image_path,model):
    img = cv2.imread(image_path,0)/255.0
    img = cv2.resize(img,(2048,256))
    img = np.expand_dims(img,0)
    result = model.predict(img)
    return result

def threshold(value):
  if value > 0.7:
    return 'True'
  else:
    return 'False'




def making_results(img_file,wav_file,model):

    filler = 'False'
    repetition = 'False'
    long_pause = 'False'

    praat_output = praat_script(wav_file)

    if praat_output > 1.0:
      result = make_prediction(img_file,model)
      result = result.flatten()
      filler,repetition = threshold(result[0]),threshold(result[1])
      filler = filler +'--'+ str(round(result[0],3))
      repetition = repetition+'--' + str(round(result[1],3))



    if praat_output < 2.0:
      long_pause = 'True'+'--'+str(praat_output)

    return ([wav_file.split('/')[-1], filler, repetition , long_pause])


def mel_spectrogram(audio_path,save_path,max_frequency):
    '''
    inputs self,save_path,frequency limits,save
    saves a image as output
    '''
    plt.figure(figsize=(14, 5))
    signal,sr = librosa.load(audio_path,sr = 22050)
    pre_emphasis = 0.97
    y = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    

    melSpec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    melSpec_dB = librosa.power_to_db(melSpec, ref=np.max)

    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge

    librosa.display.specshow(melSpec_dB, x_axis='time', y_axis='mel', sr=sr, fmax=max_frequency)

    pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
    pylab.close()

def make_spectrogram(chunked_audio_files):
    data = sorted(glob(f'{chunked_audio_files}/*.wav'))

    clean_complete_dir('/content/spectrograms')

    for no,file in enumerate(data):
        name = str(no)+'.png'
        mel_spectrogram(file,'/content/spectrograms/'+name,5000)
    
    print("making of spectrograms is done".center(100,' '))


   
def clean_complete_dir(path_to_folder):

  if not os.path.exists(path_to_folder):
    os.makedirs(path_to_folder)

  print('started cleaning directory'.center(100,' '))
  files = glob(path_to_folder+'/*')
  for f in files:
      os.remove(f)
  print('directory cleanining done'.center(100,' '))


def chunk(wav,t1,t2,newf):
    t1 = t1 * 1000 #Works in milliseconds
    t2 = t2 * 1000
    newAudio = AudioSegment.from_wav(wav)
    newAudio = newAudio[t1:t2]
    newAudio.export(newf, format="wav")

def make_chunks(audio_file):

  clean_complete_dir('/content/chunked_audio_files')


  print('initiated making chunks'.center(100,' '))
  y,sr = librosa.load(audio_file,sr = 41000)
  time_duration = librosa.get_duration(y,sr)
  timing_chunks = np.arange(0,int(time_duration),10)

  print(f'the number of chunks are {len(timing_chunks)}'.center(100,' '))

  for i in range(len(timing_chunks[:-1])):
    chunk(audio_file,timing_chunks[i],timing_chunks[i+1],f'/content/chunked_audio_files/{i}.wav')

  print('making data is done'.center(100,' '))

def resnet_model(size = (256,2048,1)):
    ''' This model is build using keras module from the paper https://arxiv.org/pdf/1910.12590.pdf
    inputs are to be resized of 256,2048,1  and the no of classification items. I have fixed to binary as default
    output is the model
    '''
    input  = Input(shape = size)
    bnEps=2e-5
    bnMom=0.9


    c1 = Conv2D(64, (7,7), padding='same',strides=2, use_bias=False,kernel_initializer='glorot_uniform')(input)
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
    m1 = GaussianNoise(0.1)(m1)
    a4 = Activation('relu')(m1)


    #-----------------------------------------------layer 2----------------------------------------------------------------------------

    c1 = Conv2D(128, (3,3),strides=2, padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a4)
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
    m1 = GaussianNoise(0.1)(m1)

    a4 = Activation('relu')(m1)

    #----------------------------------------------layer 3------------------------------------------------------------------------------

    c1 = Conv2D(128, (3,3),strides = (1,2) ,padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a4)
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
    m1 = GaussianNoise(0.1)(m1)

    a4 = Activation('relu')(m1)

    #-------------------------------------------layer 4---------------------------------------------------------------------------------

    c1 = Conv2D(64, (3,3),strides = (2,2) ,padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a4)
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
    m1 = GaussianNoise(0.1)(m1)

    a4 = Activation('relu')(m1)

    #-------------------------------------------layer 5-----------------------------------------------------------------------------------
    c1 = Conv2D(32, (3,3),strides = (2,2) ,padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a4)
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
    m1 = GaussianNoise(0.1)(m1)
    a4 = Activation('relu')(m1)

    #-----------------------------------------layer 6-------------------------------------------------------------------------
    c1 = Conv2D(16, (3,3),strides = (2,2) ,padding='same', use_bias=False,kernel_initializer='glorot_uniform')(a4)
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
    m1 = GaussianNoise(0.1)(m1)
    a4 = Activation('relu')(m1)

    f = Flatten()(a4)
    # f = Reshape((int(8192/2), 1))(f)

    # #-----------------------------------------layer7---------------------------------------------------------------------------
    # bi1 = Bidirectional(LSTM(512, return_sequences=True))(f)
    bi1 = Dense(1024,activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),activity_regularizer=l1(0.0001))(f)
    d1  = Dropout(0.2)(bi1)
    n1 = GaussianNoise(0.1)(d1)
    # model.add()

    # bi2 = Bidirectional(LSTM(512))(d1)
    bi2 = Dense(512,activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01),activity_regularizer=l1(0.0001))(n1)
    d2 = Dropout(0.4)(bi2)
    n1 = GaussianNoise(0.1)(d2)

    out = Dense(2,activation='sigmoid')(n1)

    # create model
    model = Model(inputs=input, outputs=out)
    return model

