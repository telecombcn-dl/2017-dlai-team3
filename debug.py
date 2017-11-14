import numpy as np
import cv2
import sys
import subprocess
import os
import wave
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

# img = cv2.imread('full1.0.jpg')
# x = 628
# y = 96
# w = 601
# h = 601
# crop=img[y:y+h,x:x+w]
# cv2.imshow('Video', crop)
# cv2.waitKey(0)

sample_rate, samples = wavfile.read("./output/audiovid11.0.wav")
samples = samples.sum(axis=1) / 2
frequencies, times, spectogram = signal.spectrogram(samples, sample_rate)
plt.imshow(spectogram)
plt.pcolormesh(times, frequencies, spectogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
cv2.imwrite("./output/prova.jpg", spectogram);
isface = 0