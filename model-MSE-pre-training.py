import tensorlayer as tl
import tensorflow as tf
import os
from tensorlayer.layers import *
from data_input import DataInput
from utils import norm_img, denorm_img
import argparse
from PIL import Image

DEFAULT_DATA_FACES_PATH = "/storage/dataset"
DEFAULT_DATA_AUDIOS_PATH = "/storage/dataset_videos/cropped_videos/outputb"
DEFAULT_LOG_DIR = "/storage/logs"
DEFAULT_CHECKPOINT_DIR = "/storage/checkpoints"


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from tensorflow.python.client import device_lib
print device_lib.list_local_devices()

def lrelu1(x, name="lrelu1"):
    return tf.maximum(x, 0.25*x)

def restore_model(sess, checkpoint_path):
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)

    if checkpoint_path is not None:
        restorer = tf.train.Saver()
        restorer.restore(sess, ckpt.model_checkpoint_path)


# TODO: ADD SKIP CONNECTIONS (To improve performance, not in the original began paper)
def generator(input_audio, reuse, hidden_number=64, kernel=3):
    w_init = tf.random_normal_initializer(stddev=0.02)

    with tf.variable_scope("generator_MSE", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        # EXTRACT AUDIO FEATURES
        x = InputLayer(input_audio, name="in_audio_features_extractor") #[batch_size, height, width, 1]
        x = Conv2dLayer(x, shape=[kernel, kernel, 1, 64], strides=[1, 1, 1, 1], padding='SAME', W_init=w_init,
                        act=lrelu1, name='AudioFeatures/conv1')
        x = Conv2dLayer(x, shape=[kernel, kernel, 64, 128], strides=[1, 1, 1, 1], padding='SAME', W_init=w_init,
                        act=lrelu1, name='AudioFeatures/conv2')
        # max o avg pool?
        # stride only time axis (ESTA BIEN?)
        x = PoolLayer(x, strides=[1, 2, 1, 1], pool=tf.nn.avg_pool, name='AudioFeatures/pool1')
        x = Conv2dLayer(x, shape=[kernel, kernel, 128, 256], strides=[1, 1, 1, 1], padding='SAME', W_init=w_init,
                        act=lrelu1, name='AudioFeatures/conv3')
        x = Conv2dLayer(x, shape=[kernel, kernel, 256, 512], strides=[1, 1, 1, 1], padding='SAME', W_init=w_init,
                        act=lrelu1, name='AudioFeatures/conv4')
        x = PoolLayer(x, strides=[1, 2, 1, 1], pool=tf.nn.avg_pool, name='AudioFeatures/pool2')
        x = FlattenLayer(x, name='AudioFeatures/flatten')
        x = DenseLayer(x, n_units=512, name='AudioFeatures/dense1', act=lrelu1)
        x = DenseLayer(x, n_units=256, name='AudioFeatures/dense2', act=lrelu1) #[batch_size, 256]

        # DECODER BEGINS
        # hidden_number = n = 128
        # exponential linear units output convolutions
        # Each layer is repeated a number of times (typically 2). We observed that more repetitions led to
        # even better visual results
        # Down-sampling is implemented as sub-sampling with stride 2 and up- sampling is done by nearest neighbor.
        x = DenseLayer(x, n_units=8*8*hidden_number, name='Generator_MSE/dense2')
        arguments = {'shape': [-1, 8, 8, hidden_number], 'name': 'Generator_MSE/reshape1'}
        x = LambdaLayer(x, fn=tf.reshape, fn_args=arguments)
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1,1,1,1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu,name='Generator_MSE/conv1')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu,name='Generator_MSE/conv2')
        x = UpSampling2dLayer(x, size=[2, 2], is_scale=True, method=1, name='Generator_MSE/UpSampling1') # method= 1 NN

        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu,name='Generator_MSE/conv3')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Generator_MSE/conv4')
        x = UpSampling2dLayer(x, size=[2, 2], is_scale=True, method=1, name='Encoder_MSE/UpSampling2')  # method= 1 NN

        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu,name='Generator_MSE/conv5')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu,name='Generator_MSE/conv6')
        x = UpSampling2dLayer(x, size=[2, 2], is_scale=True, method=1, name='Generator_MSE/UpSampling3')  # method= 1 NN

        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1],
                        padding='SAME',
                        W_init=w_init,act=tf.nn.elu, name='Generator_MSE/conv7')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Generator_MSE/conv8')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, 3], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, name='Generator_MSE/convLAST')

        return x


def discriminator(disc_input, reuse, z_num=64, hidden_number=128, kernel=3):
    w_init = tf.random_normal_initializer(stddev=0.02)

    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        # Encoder
        # Down-sampling is implemented as sub-sampling with stride 2

        x = InputLayer(disc_input, name='in')  # [1, height = 64, width = 64, 3 ]
        x = Conv2dLayer(x, shape=[kernel, kernel, 3, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init,act=tf.nn.elu, name='Discriminator/Encoder/conv1')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init,act=tf.nn.elu, name='Discriminator/Encoder/conv2')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, 2*hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init,act=tf.nn.elu, name='Discriminator/Encoder/conv3')
        x = Conv2dLayer(x, shape=[kernel, kernel, 2*hidden_number, 2*hidden_number], strides=[1, 2, 2, 1], padding='SAME',
                        W_init=w_init,act=tf.nn.elu, name='Discriminator/Encoder/subsampling1')
        # [1, height = 32, width = 32, 2*hidden_number]

        x = Conv2dLayer(x, shape=[kernel, kernel, 2*hidden_number, 2*hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init,act=tf.nn.elu, name='Discriminator/Encoder/conv4')
        x = Conv2dLayer(x, shape=[kernel, kernel, 2*hidden_number, 3 * hidden_number], strides=[1, 1, 1, 1],
                        padding='SAME', W_init=w_init, act=tf.nn.elu, name='Discriminator/Encoder/conv5')
        x = Conv2dLayer(x, shape=[kernel, kernel, 3 * hidden_number, 3 * hidden_number], strides=[1, 2, 2, 1],
                        padding='SAME', W_init=w_init, act=tf.nn.elu, name='Discriminator/Encoder/subsampling2')
        # [1, height = 16, width = 16, 3*hidden_number]

        x = Conv2dLayer(x, shape=[kernel, kernel, 3 * hidden_number, 3 * hidden_number], strides=[1, 1, 1, 1],
                        padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Discriminator/Encoder/conv6')
        x = Conv2dLayer(x, shape=[kernel, kernel, 3 * hidden_number, 4 * hidden_number], strides=[1, 1, 1, 1],
                        padding='SAME', W_init=w_init, act=tf.nn.elu, name='Discriminator/Encoder/conv7')
        x = Conv2dLayer(x, shape=[kernel, kernel, 4 * hidden_number, 4 * hidden_number], strides=[1, 2, 2, 1],
                        padding='SAME', W_init=w_init, act=tf.nn.elu, name='Discriminator/Encoder/subsampling3')
        # [1, height = 8, width = 8, 4*hidden_number]

        x = Conv2dLayer(x, shape=[kernel, kernel, 4 * hidden_number, 4 * hidden_number], strides=[1, 1, 1, 1],
                        padding='SAME',  W_init=w_init, act=tf.nn.elu, name='Discriminator/Encoder/conv8')
        x = Conv2dLayer(x, shape=[kernel, kernel, 4 * hidden_number, 4 * hidden_number], strides=[1, 1, 1, 1],
                        padding='SAME', W_init=w_init, act=tf.nn.elu, name='Discriminator/Encoder/conv9')

        x = FlattenLayer(x, name='Discriminator/Encoder/flatten')
        z = DenseLayer(x, n_units=z_num, name='Discriminator/Encoder/Dense')

        # Decoder
        x = DenseLayer(x, n_units=8 * 8 * hidden_number, name='Generator/dense2')
        arguments = {'shape': [-1, 8, 8, hidden_number], 'name': 'Generator/reshape1'}
        x = LambdaLayer(x, fn=tf.reshape, fn_args=arguments)
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Generator/conv1')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Generator/conv2')
        x = UpSampling2dLayer(x, size=[2, 2], is_scale=True, method=1, name='Generator/UpSampling1')  # method= 1 NN

        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Generator/conv3')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Generator/conv4')
        x = UpSampling2dLayer(x, size=[2, 2], is_scale=True, method=1, name='Encoder/UpSampling2')  # method= 1 NN

        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Generator/conv5')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Generator/conv6')
        x = UpSampling2dLayer(x, size=[2, 2], is_scale=True, method=1, name='Generator/UpSampling3')  # method= 1 NN

        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1],
                        padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Generator/conv7')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Generator/conv8')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, 3], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, name='Generator/convLAST')

        return x, z


def train(batch_size, epochs, dataset, log_dir):
    image_width = 64
    image_height = 64
    audio_height = 35
    audio_width = 11

    # ##========================== DEFINE INPUT DATA ============================###
    images = tf.placeholder('float32', [None, image_height, image_width, 3], name='t_image_generator')
    audio = tf.placeholder('float32', [None, audio_height, audio_width, 1], name='t_audio_input_generator')
    tf.summary.image('input_image', images)
    tf.summary.image('audio', audio)

    # ##========================== DEFINE MODEL ============================###
    net_gen = generator(input_audio=audio, reuse=False)
    tf.summary.image('norm_generated_image', net_gen.outputs)
    tf.summary.image('generated_image', denorm_img(net_gen.outputs))

    output_gen = denorm_img(net_gen.outputs)  # Denormalization

    g_vars = tl.layers.get_variables_with_name('generator', True, True)

    lr = tf.Variable(0.00004, trainable=False)

    g_MSE = tf.reduce_mean(tf.square(images - output_gen), name='g_loss_gan')

    g_MSE_optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(g_MSE, var_list=g_vars)
    tf.summary.scalar('MSE_loss', g_MSE)

    summary = tf.summary.merge_all()
    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=1)
        # Summary writer to save logs
        summary_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'), sess.graph)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        if args.resume == "True":
            print("Restoring model from checkpoint")
            restore_model(sess, checkpoint_path=args.checkpoint_dir)

        items_faces, items_audio = dataset.get_items()
        total = 0
        for j in range(0, epochs):
            iteration = 0
            while iteration * batch_size < len(items_faces):
                input_images = np.empty([batch_size, 64, 64, 3])
                audio_MFCC = np.empty([batch_size, 35, 11, 1])
                count = 0
                for face, input_audio in zip(items_faces[iteration * batch_size:iteration * batch_size + batch_size],
                                             items_audio[iteration * batch_size:iteration * batch_size + batch_size]):
                    input_image = Image.open(face)
                    input_image = np.asarray(input_image, dtype=float)
                    input_images[count] = input_image
                    input_audio = np.load(input_audio)
                    input_audio = np.asarray(input_audio, dtype=float)
                    audio_MFCC[count] = input_audio[:, :, np.newaxis]
                    count += 1
                # ##========================= train BEGAN =========================###
                _, summary_str, MSE_loss = sess.run([g_MSE_optim, summary, g_MSE], feed_dict={images: input_images,
                                                                                              audio: audio_MFCC})
                print("PRE-TRAINING GENERATOR -- Epoch: {} Iteration: {} MSE_loss: {}.".format(j, iteration, MSE_loss))
                summary_writer.add_summary(summary_str, total)

                # ##========================= save checkpoint =========================###
                if iteration % 3000 == 0 and iteration > 0:
                    tf.logging.info('Saving checkpoint')
                    saver.save(sess, args.checkpoint_dir + "/checkpoint", global_step=iteration, write_meta_graph=False)
                iteration += 1
                total += 1
            rest = len(items_faces) - ((iteration - 1)*batch_size)
            if rest > 0:
                count = 0
                input_images = np.empty([rest, 64, 64, 3])
                audio_MFCC = np.empty([rest, 35, 11, 1])
                for face, input_audio in zip(items_faces[len(items_faces)-rest:], items_audio[len(items_faces)-rest:]):
                    input_image = Image.open(face)
                    input_image = np.asarray(input_image, dtype=float)
                    input_images[count] = input_image
                    input_audio = np.load(input_audio)
                    input_audio = np.asarray(input_audio, dtype=float)
                    audio_MFCC[count] = input_audio[:, :, np.newaxis]
                    count += 1
                # ##========================= train BEGAN =========================###
                _, summary_str, MSE_loss = sess.run([g_MSE_optim, summary, g_MSE], feed_dict={images: input_images,
                                                                                              audio: audio_MFCC})
                print("PRE-TRAINING GENERATOR -- Epoch: {} Iteration: {} MSE_loss: {}.".format(j, iteration, MSE_loss))
                summary_writer.add_summary(summary_str, total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict script')
    parser.add_argument('-dataset_faces_folder', default=DEFAULT_DATA_FACES_PATH, help='Path to the images file')
    parser.add_argument('-dataset_audios_folder', default=DEFAULT_DATA_AUDIOS_PATH, help='Path to the audios file')
    parser.add_argument('-checkpoint_dir', default=DEFAULT_CHECKPOINT_DIR, help='Model checkpoint to use')
    parser.add_argument('-log_dir', default=DEFAULT_LOG_DIR, help='Model checkpoint to use')
    parser.add_argument('-resume', default="True", help='Resume training ("True" or "False")')

    args = parser.parse_args()

    if args.resume == "False":
        if tf.gfile.Exists(args.log_dir):
            tf.gfile.DeleteRecursively(args.log_dir)
        tf.gfile.MakeDirs(args.log_dir)

    if not os.path.isdir(os.path.dirname(args.checkpoint_dir)):
        os.mkdir(os.path.dirname(args.checkpoint_dir))

    train(batch_size=16, epochs=20, dataset=DataInput(args.dataset_faces_folder,
                                                      args.dataset_audios_folder, "train"), log_dir=args.log_dir)
