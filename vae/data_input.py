from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

DATA_DIRECTORY = "data"

# Params for MNIST
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.

DEFAULT_DATA_AUDIOS_PATH = "/storage/dataset_videos/cropped_videos/outputb"


def get_items():
    audio_list = [os.path.join(DEFAULT_DATA_AUDIOS_PATH, f) for f in os.listdir(DEFAULT_DATA_AUDIOS_PATH)
                  if os.path.isfile(os.path.join(DEFAULT_DATA_AUDIOS_PATH, f)) and
                  '_MFCC2_' in f]
    return audio_list


def input_images_audios():
    items_audio = get_items()
    print("Dataset size is: {}".format(len(items_audio)))
    audios = np.empty([len(items_audio), 35*11])
    count = 0
    for audio in items_audio:
        audio = np.load(audio)
        audio = np.asarray(audio, dtype=float)
        # Normalize the values so they go from 0 to 1
        audio = np.clip(audio, -39.99999, 39.99999)
        audio = (audio + 40) / 80
        # Reshape into a vector of size 35*12
        audios[count] = audio.reshape([1, 35*11])
        count += 1
    return audios


# Prepare MNISt data
def prepare_MFCC_dataset():
    train_total_data = input_images_audios()
    train_size = train_total_data.shape[0]
    return train_total_data, train_size
