import numpy as np
import cv2
import sys
import subprocess
import os
import wave
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav

def format_time(msecs):
    #Segurament es podira optimitzar fent operacions de modul
    #en comptes de divisions i casts
    h = int(msecs/3600000)
    m = int((msecs/60000) - (h*60))
    s = int((msecs/1000) - (m*60))
    ms = int(msecs - s*1000)
    return h,m,s,ms

audio_lenght = 35
audio_lenght2 = 65
face_margin = 1.25

if not os.path.exists('./output'):
    os.makedirs('./output')

with open('vid_list.txt') as f:
    for line in f:
        print(line)
        next_video = line.rstrip('\n')
        input_vid = next_video
        video_capture = cv2.VideoCapture(input_vid+'.mp4')

        command = "ffmpeg -i " + input_vid + ".mp4 -vn "+input_vid+".wav"
        subprocess.call(command, shell=True)
        rate, sig = wavfile.read(input_vid+".wav")
        sig = sig.sum(axis=1) / 2  # per passar a monochanel en principi
        audio_lenght_n = int(round(audio_lenght*rate/1000))
        audio_lenght2_n = int(round(audio_lenght2*rate/1000))

        width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
        height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = num_frames/fps
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 2)
        current_frame = 2
        while (current_frame < num_frames-100): #deixem un marge de 100 frames per si mal codificat
            current_frame = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
            current_time = video_capture.get(cv2.CAP_PROP_POS_MSEC)
            video_capture.grab()
            current_time_audio_sample=int(round(current_time*rate/1000))
            aux = current_time_audio_sample-int(round(audio_lenght_n/2))
            audio_seg = sig[current_time_audio_sample-int(round(audio_lenght_n/2)):current_time_audio_sample+int(round(audio_lenght_n/2))]
            audio_seg2 = sig[current_time_audio_sample-int(round(audio_lenght2_n/2)):current_time_audio_sample+int(round(audio_lenght2_n/2))]
            mfcc_feat = mfcc(audio_seg, rate,0.002, 0.001)
            mfcc_feat = mfcc_feat[:,1:]
            mfcc_feat2 = mfcc(audio_seg2, rate,0.004, 0.002)
            mfcc_feat2 = mfcc_feat[:,1:]
            cv2.imwrite("./output/" + input_vid + "_spectogram_" + str(current_frame) + ".jpg", mfcc_feat);
            np.save('./output/' + input_vid + '_MFCC_' + str(current_frame), mfcc_feat)
            cv2.imwrite("./output/" + input_vid + "_spectogram2_" + str(current_frame) + ".jpg", mfcc_feat2);
            np.save('./output/' + input_vid + '_MFCC2_' + str(current_frame), mfcc_feat2)

        video_capture.release()
