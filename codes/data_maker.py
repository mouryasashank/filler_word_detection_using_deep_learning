import os
import numpy as np
import librosa
from pydub import AudioSegment
from tqdm import tqdm
from glob import glob

def min2sec(t):
    mins,secs = t.split('.')
    return (60*int(mins)+int(secs))

def get_timings(d):
    timings = []
    name = d[0]
    d = d[1:]
    for j in d:
        if len(j.split('-')) > 2:
            t1,t2,cat = j.split('-')
            t1_sec,t2_sec = min2sec(t1),min2sec(t2)
            timings.append([t1_sec,t2_sec,cat])
        else:
            print(name+'.wav','-'.center(100,'-'))
            print('less than two variables',j)        
    return timings


def chunk(wav,t1,t2,newf):
    t1 = t1 * 1000 #Works in milliseconds
    t2 = t2 * 1000
    newAudio = AudioSegment.from_wav(wav)
    newAudio = newAudio[t1:t2]
    newAudio.export(newf, format="wav")

def audio_data_making(txtfile,files,label_name,root_dir='/content/drive/My Drive/'):
    f  = open(txtfile)
    data = f.read()
    audio_dict = dict()
    for i in data.split('\n'):
        d = i.split(',')
        try:
            t = get_timings(d)
        except:
            print('some error in for loop')
            
        audio_dict[root_dir+label_name+'/'+d[0]+'.wav'] = t

    # print(audio_dict)
    try:
        os.mkdir('chunks')
    except:
        print('chunks directory is already present')

    for aud in tqdm(files):
        name = aud.split('/')[-1].split('.wav')[0]
              
        
        if len(audio_dict[root_dir+label_name+'/'+name+'.wav']) > 0:
            y,sr = librosa.load(aud,sr = 41000)
            time_duration = librosa.get_duration(y,sr)
            for t1 in range(int(time_duration)-10):
                t2 = t1+10
                category = '-'
                for j in audio_dict[root_dir+label_name+'/'+name+'.wav']:
                    tt1,tt2,cat = j
                    if tt1-t1>0 and tt2-t2<0:
                        category = category+','+cat
                chunk(aud,t1,t1+10,'chunks/'+name+'-'+str(t1)+'-'+str(t2)+category+'-'+'.wav')
        else:
            y,sr = librosa.load(aud,sr = 41000)
            time_duration = librosa.get_duration(y,sr)
            for t1 in range(int(time_duration)-10):
                t2 = t1+10
                category = '-'
                chunk(aud,t1,t1+10,'chunks/'+name+'-'+str(t1)+'-'+str(t2)+category+'-'+'.wav')
