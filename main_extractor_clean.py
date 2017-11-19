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

min_face_size_relative = 0.15
audio_lenght = 35 #less than 999ms
face_margin = 1.25

if not os.path.exists('./output'):
    os.makedirs('./output')

file_association = open('./output/associations.txt','w')

with open('vid_list.txt') as f:
    for line in f:
        print(line)
        next_video = line.rstrip('\n')
        input_vid = next_video
        faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#Haar filter definition
        video_capture = cv2.VideoCapture(input_vid+'.mp4')#mp4 format assumet for all videos
        width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
        height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = num_frames/fps
        min_face_size = int(height*min_face_size_relative)
        print(min_face_size)


        #preview
        nsteps = 40
        step_preview = int(num_frames/nsteps)
        num_previews = 0
        dimensions = 0
        for i in range(0,num_frames-100,step_preview):
            print(i)
            video_capture.set(cv2.CAP_PROP_POS_FRAMES,i)
            current_frame = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(min_face_size, min_face_size),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces)==1:
                face = faces[0]
                num_previews = num_previews +1
                dimensions = dimensions + face[2]
        dimensions = dimensions/num_previews
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 600)
        current_frame = 600

        while (current_frame < num_frames-200): #deixem un marge de 100 frames per si mal codificat
            current_frame = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
            current_time = video_capture.get(cv2.CAP_PROP_POS_MSEC)
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(min_face_size, min_face_size),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            frame_ori = frame.copy()
            if len(faces) == 1:
                face = faces[0]
                x = face[0]
                y = face[1]
                x_start = int(x-dimensions*(face_margin-1))
                x_end = int(x+dimensions*face_margin)
                y_start=int(y-dimensions*(face_margin-1))
                y_end = int(y+dimensions*face_margin)

                if y_start<0:
                    y_start=0
                    y_end = int(y_start + dimensions*face_margin)
                if y_end>height:
                    y_end = int(height)
                    y_start = int(y_end -dimensions*face_margin)
                if x_start<0:
                    x_start=0
                    x_end = int(x_start + dimensions*face_margin)
                if x_end>width:
                    x_end = int(width)
                    x_start = int(x_end - dimensions*face_margin)

                croped_face = frame_ori[y_start:y_end,x_start:x_end]
                croped_face_resized = cv2.resize(croped_face, (64, 64))
                #cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                cv2.imwrite("./output/"+input_vid+"_face_"+str(current_frame)+".jpg", croped_face_resized);
                t_start = current_time - audio_lenght/2
                t_start_h,t_start_m,t_start_s,t_start_ms = format_time(t_start)
                str_start_time = str(t_start_h).zfill(2)+":"+str(t_start_m).zfill(2)+":"+str(t_start_s).zfill(2)+"."+str(int(t_start_ms)).zfill(3)
                command = "ffmpeg -i "+input_vid+".mp4"+" -ss "+str_start_time+" -t 00:00:00."+str(audio_lenght).zfill(3)+" -ab 160k -ac 2 -ar 44100 -vn ./output/audio"+input_vid+str(current_frame)+".wav"
                subprocess.call(command, shell=True)
                rate, sig = wavfile.read("./output/audio"+input_vid+str(current_frame)+".wav")
                sig = sig.sum(axis=1) / 2 #per passar a monochanel en principi
                mfcc_feat = mfcc(sig, rate,0.002, 0.001)
                mfcc_feat = mfcc_feat[:,1:]
                file_association.write(input_vid+"_face_"+str(current_frame)+".jpg  " + input_vid + '_MFCC_' + str(current_frame) +'.npy\n')
                cv2.imwrite("./output/" + input_vid + "_spectogram_" + str(current_frame) + ".jpg", mfcc_feat);
                np.save('./output/' + input_vid + '_MFCC_' + str(current_frame), mfcc_feat)
            print(current_frame)

            #cv2.imshow('Video', frame)
            #cv2.waitKey(1)


        video_capture.release()
        cv2.destroyAllWindows()
    file_association.close()