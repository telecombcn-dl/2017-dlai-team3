import tensorflow as tf
import os
from PIL import Image
import numpy as np

AUDIO_WIDTH = 12
AUDIO_HEIGHT = 35
path = "/storage/dataset_videos/audio2faces_dataset/"

def get_input_items(iteration, batch_size, path):
    dataset = DataInput(data_path, "train")




def _read_item(face_queue, audio_queue):
    reader = tf.WholeFileReader()
    _, face_image = reader.read(face_queue)
    #_, audio_image = reader.read(audio_queue)
    _, audio_MFCC = reader.read(audio_queue)


    with tf.name_scope('decode_face_image'):
        face_image = tf.image.decode_jpeg(face_image, channels=3)
        face_image = tf.to_float(face_image)
        face_image = tf.reshape(face_image, [64, 64, 3])

    # with tf.name_scope('decode_audio_image'):
    #     audio_image = tf.image.decode_jpeg(audio_image, channels=1)
    #     audio_image = tf.to_float(audio_image)
    #     audio_image = tf.reshape(audio_image, [AUDIO_HEIGHT, AUDIO_WIDTH, 1])

    with tf.name_scope('decode_audio_MFCC'):
        audio_MFCC = tf.decode_raw(audio_MFCC, out_type=tf.float64)
        audio_MFCC = tf.reshape(audio_MFCC, [AUDIO_HEIGHT, AUDIO_WIDTH, 1])

    print(audio_MFCC.shape)
    return face_image, audio_MFCC


def _create_batch(example, batch_size, num_threads):
    # Specify the samples queue parameters
    min_after_dequeue = 10 * batch_size
    capacity = min_after_dequeue + (num_threads + 3) * batch_size
    return tf.train.batch(example, batch_size=batch_size, capacity=capacity, num_threads=num_threads)


class DataInput(object):

    def __init__(self, path, name):
        self.path = path
        self.name = name

    def input_pipeline(self, batch_size, num_epochs, shuffle=False, num_threads=4):
        with tf.device('/cpu:0'):
            with tf.name_scope(self.name):
                items_faces, items_audio = self._get_input_queue_items()
                face_queue = tf.train.string_input_producer(items_faces, num_epochs=num_epochs, shuffle=shuffle)
                audio_queue = tf.train.string_input_producer(items_audio, num_epochs=num_epochs, shuffle=shuffle)
                example = _read_item(face_queue, audio_queue)
                return _create_batch(example, batch_size, num_threads)

    def get_items(self):
        print("Called input queue")
        face_image_list = [os.path.join(self.path, f) for f in os.listdir(self.path)
                           if os.path.isfile(os.path.join(self.path, f)) and
                           '_face_' in f]

        audio_image_list = [(item.replace("_face_", "_MFCC_")) for item in face_image_list]
        audio_image_list = [(item.replace(".jpg", ".npy")) for item in audio_image_list]
        # print(len(face_image_list))
        # print(len(face_image_list)/16)
        # print(int(len(face_image_list) / 16))

        return face_image_list, audio_image_list


if __name__ == '__main__':

    data_path = "/storage/dataset"

    dataset = DataInput(data_path, "train")

    face_image, audio_MFCC = dataset.input_images_audios(batch_size=1, iteration=0)
    print(face_image[0].shape)
    print(audio_MFCC[0].shape)

    # face_image, audio_MFCC = dataset.input_pipeline(batch_size=1, num_epochs=1, shuffle=False)
    #
    # with tf.Session() as sess:
    #     init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #     sess.run(init_op)
    #
    #     # Coordinate the different workers for the input data pipeline
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #
    #     for i in range(1):
    #         face_image_value, audio_MFCC = sess.run([face_image, audio_MFCC])
    #         print face_image_value[0].shape
    #         print audio_MFCC[0]
    #
    #     coord.request_stop()
    #     coord.join(threads)
