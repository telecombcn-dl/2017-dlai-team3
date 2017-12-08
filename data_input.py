import tensorflow as tf
import os
from PIL import Image
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np

AUDIO_WIDTH = 12
AUDIO_HEIGHT = 35


def _read_item(face_queue, audio_queue):
    reader = tf.WholeFileReader()
    _, face_image = reader.read(face_queue)
    _, audio_MFCC = reader.read(audio_queue)

    with tf.name_scope('decode_face_image'):
        face_image = tf.image.decode_jpeg(face_image, channels=3)
        face_image = tf.to_float(face_image)
        face_image = tf.reshape(face_image, [64, 64, 3])

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

    def __init__(self, path_faces, path_audio, name):
        self.path_faces = path_faces
        self.path_audio = path_audio
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
        image_list = [os.path.join(self.path_faces, f) for f in os.listdir(self.path_faces)
                      if os.path.isfile(os.path.join(self.path_faces, f)) and
                      '_face_' in f]

        audio_list = [(item.replace(self.path_faces, self.path_audio)) for item in image_list]
        audio_list = [(item.replace("_face_", "_MFCC2_")) for item in audio_list]
        audio_list = [(item.replace(".jpg", ".npy")) for item in audio_list]

        return image_list, audio_list

    def input_images_audios(self, batch_size, iteration):
        print("Called input images audio")
        items_faces, items_audio = self.get_items()
        print(len(items_faces))
        faces = np.empty([batch_size, 64, 64, 3])
        audios = np.empty([batch_size, 35, 12, 1])
        count = 0
        for face, audio in zip(items_faces[iteration*batch_size:iteration*batch_size+batch_size],
                               items_audio[iteration*batch_size:iteration*batch_size+batch_size]):
            image = Image.open(face)
            image = np.asarray(image, dtype=float)
            faces[count] = image
            audio = np.load(audio)
            audio = np.asarray(audio, dtype=float)
            audios[count] = audio[:, :, np.newaxis]
            count += 1
        return faces, audios


if __name__ == '__main__':
    path = "/storage/dataset_videos/audio2faces_dataset/"
    data_path_faces = "/storage/dataset"
    data_path_audios = "/storage/dataset_videos/cropped_videos/outputb"

    dataset = DataInput(data_path_faces, data_path_audios, "train")

    face_image_list, audio_MFCC_list = dataset.get_items()

    print("_______________")
    print(len(face_image_list))
    print(len(audio_MFCC_list))
    face_image, audio_MFCC = dataset.input_images_audios(batch_size=2, iteration=0)

    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        # Coordinate the different workers for the input data pipeline
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        audio = tf.placeholder('float32', [None, 35, 12, 1],
                               name='t_audio_input_generator')

        for i in range(1):
            x = InputLayer(audio, name="in_audio_features_extractor")
            x = FlattenLayer(x, name='AudioFeatures/flatten')
            audio_features = DenseLayer(x, n_units=512, name='AudioFeatures/dense1')
            audio_concat = tf.concat([audio_features.outputs, tf.random_uniform(shape=[1, 256], minval=-1, maxval=1)], axis=1)
            x = InputLayer(audio_concat, name="in_audio_features")
            face_image_value, audio_MFCC, audio_concat = sess.run([face_image, audio_MFCC, audio_concat],
                                                                  feed_dict={audio: audio_MFCC})

            print face_image_value[0].shape
            print audio_MFCC.shape[0]
            #print audio_concat.shape[0]

        # coord.request_stop()
        # coord.join(threads)
