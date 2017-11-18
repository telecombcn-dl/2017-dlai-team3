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

#parameters

#Indiquem el tamany relatiu respecte les dimensions del video que han de tenir les cares detectades
min_face_size_relative = 0.15
#Indiquem quin es la longitud del segments d'audio que volem extreure
audio_lenght = 35 #si supera els 999 ms s'hauria de modificar el codi
#Indiquem el percentatge d'overlap que volem entre segments d'audio
overlap=0.25

#Crem la carpeta de outpu en cas de que no existeixi
if not os.path.exists('./output'):
    os.makedirs('./output')

#Obrim el fitxer de text que conte el nom de tots els videos a processar
#i comencem a processar-los un per un
with open('vid_list.txt') as f:
    for line in f:

        print(line)
        next_video = line.rstrip('\n')

        input_vid = next_video
        #Carreguem les filtres de haar del fitxer especificat
        #Actualment s'utiltiza el un de generic, potser en podem trobar algun de millor
        #prer internet n'hi han varis
        faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        #Assumeixo que tots els videos estan en format mp4, en cas contrari s'hauria de fer
        #algunes modificaciosn
        video_capture = cv2.VideoCapture(input_vid+'.mp4')
        width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
        height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = num_frames/fps
        print(video_duration)
        print(num_frames)
        print(width)
        print(height)
        print(fps)
        #Calculem el tamany de cara minima en pixels
        min_face_size = int(height*min_face_size_relative)
        print(min_face_size)
        #Calculem la longitud d'audio en frames
        audio_lenght_frames = int(round(fps*audio_lenght/1000))
        overlap_frames = int(round(fps*overlap*audio_lenght/1000))

        #Acabar e tenir en compte overlap, quina es la fram mitat

        start_win_frame=0

        while True:
            # Capture frame-by-frame
            ready_capture = 1
            current_frame = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            current_time = video_capture.get(cv2.CAP_PROP_POS_MSEC)
            print(current_time)

            if (ready_capture):
                #El programa esta disenyat per anar calculant les finestres d'audio a temps real
                #Potser seria millor segmentar el video abans de començar a processar res per
                #tal de millorar l'eficencia

                end_win_frame = start_win_frame + audio_lenght_frames
                if end_win_frame > num_frames:
                    break
                midle_win_frame = start_win_frame+round((end_win_frame - start_win_frame)/2)
                ready_capture = 0

            #En cas que la frame actual no sigui la frame que volem guardar la passem sense processar
            if current_frame!=midle_win_frame:
                video_capture.grab()
            #En cas que la frame actual sigui la frame que volem guardar extraiem la cara
            else:
                start_win_frame = end_win_frame-overlap_frames
                ready_capture = 1
                ret, frame = video_capture.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(min_face_size, min_face_size),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                isface = 0
                # Draw a rectangle around the faces
                frame_ori = frame.copy()
                #En aquest for recorrem totes les frames detectades
                for (x, y, w, h) in faces:
                    isface = 1
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                #Podria ser bona idea agafar la cara més centrada/tamany bo i no només la primera
                #Segurament també seria bona idea tenir en compte dimensions totals per no sortir-se de la imatge si fem
                #extensio del rectangle
                if isface:
                    #Extraiem la cara detectada, si volem que el cropping no sigui tan agresiu podriem extendre
                    #el rectangle de detecio per agafar una zona més gran del plan:
                    #croped_face = frame_ori[y-algo:y+h+alog,x-algo:x+w+algo]
                    croped_face = frame_ori[y:y+h,x:x+w]
                    #guardem al imatge
                    cv2.imwrite("./output/"+input_vid+"_face_"+str(current_frame)+".jpg", croped_face);
                    current_time_h,current_time_m,current_time_s,current_time_ms = format_time(current_time)
                    if (current_time + audio_lenght)/1000 > video_duration:
                        break
                    #guardem l'audio
                    str_start_time = str(current_time_h).zfill(2)+":"+str(current_time_m).zfill(2)+":"+str(current_time_s).zfill(2)+"."+str(int(current_time_ms))
                    command = "ffmpeg -i "+input_vid+".mp4"+" -ss "+str_start_time+" -t 00:00:00."+str(audio_lenght).zfill(3)+" -ab 160k -ac 2 -ar 44100 -vn ./output/audio"+input_vid+str(current_frame)+".wav"
                    print(command)
                    subprocess.call(command, shell=True)

                    #Intentem crear spectograma pel segment extret
                    #Potser no hauriem d'extreure segment a segment sino tot el tros i llavors
                    #crear espectogrames per cada segment?
                    rate, sig = wavfile.read("./output/audio"+input_vid+str(current_frame)+".wav")
                    sig = sig.sum(axis=1) / 2 #per passar a monochanel en principi


                    mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
                             nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97,
                             ceplifter=22, appendEnergy=True)
                    mfcc_feat = mfcc(sig, rate,)
                    d_mfcc_feat = delta(mfcc_feat, 2)
                    fbank_feat = logfbank(sig, rate)

                    print(fbank_feat[1:3, :])

                    cv2.imwrite("./output/" + input_vid + "_spectogram_" + str(current_frame) + ".jpg", fbank_feat);
                    isface = 0

                    # Display the resulting frame

                #El programa esta disenyat perque mostri el video i les cares a mesura que es van processant per temes de debug
                #En la versio final s'hauria de suprimir imshow i no caldria guarda la frame com a frame_ori
                cv2.imshow('Video', frame)
                cv2.waitKey(1)
            #cv2.waitKey(1) & 0xFF == ord('q'):
                #break

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()
