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


def restore_model(sess):
    # define model
    t_input_gen = tf.placeholder('float32', [None, 35, 12, 1], name='t_image_input_generator')
    netw = generator(t_input_gen, reuse=True)
    # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    saver = tf.train.Saver(tf.trainable_variables(scope="generator"))
    # saver.restore(sess, DEFAULT_CHECKPOINT_DIR)

    # Get the state of the checkpoint and then restore using ckpt path
    ckpt = tf.train.get_checkpoint_state(args.checkpoint_dir)

    if args.checkpoint_dir is not None:
        restorer = tf.train.Saver()
        print(ckpt.model_checkpoint_path)
        restorer.restore(sess, '/storage/checkpoints/checkpoint-4')


# TODO: ADD SKIP CONNECTIONS (To improve performance, not in the original began paper)
def generator(input_audio, reuse, hidden_number=64, kernel=3):
    w_init = tf.random_normal_initializer(stddev=0.02)

    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        # EXTRACT AUDIO FEATURES
        x = InputLayer(input_audio, name="in_audio_features_extractor") #[batch_size, height, width, 1]
        x = Conv2dLayer(x, shape=[kernel, kernel, 1, 64], strides=[1, 1, 1, 1], padding='SAME', W_init=w_init,
                        name='AudioFeatures/conv1')
        x = Conv2dLayer(x, shape=[kernel, kernel, 64, 128], strides=[1, 1, 1, 1], padding='SAME', W_init=w_init,
                        name='AudioFeatures/conv2')
        # max o avg pool?
        # stride only time axis (ESTA BIEN?)
        x = PoolLayer(x,strides=[1, 2, 1, 1], pool=tf.nn.avg_pool, name='AudioFeatures/pool1')
        x = Conv2dLayer(x, shape=[kernel, kernel, 128, 256], strides=[1, 1, 1, 1], padding='SAME', W_init=w_init,
                        name='AudioFeatures/conv3')
        x = Conv2dLayer(x, shape=[kernel, kernel, 256, 512], strides=[1, 1, 1, 1], padding='SAME', W_init=w_init,
                        name='AudioFeatures/conv4')
        x = PoolLayer(x, strides=[1, 2, 1, 1], pool=tf.nn.avg_pool, name='AudioFeatures/pool2')
        x = FlattenLayer(x, name='AudioFeatures/flatten')
        x = DenseLayer(x, n_units=512, name='AudioFeatures/dense1')
        audio_features = DenseLayer(x, n_units=256, name='AudioFeatures/dense2') #[batch_size, 256]

        # DECODER BEGINS
        # hidden_number = n = 128
        # exponential linear units output convolutions
        # Each layer is repeated a number of times (typically 2). We observed that more repetitions led to
        # even better visual results
        # Down-sampling is implemented as sub-sampling with stride 2 and up- sampling is done by nearest neighbor.
        input_generator = tf.concat([audio_features.outputs, tf.random_uniform(
            shape=[tf.shape(audio_features.outputs)[0], 256], maxval=1, minval=-1)], axis=1)
        x = InputLayer(input_generator, name="in")
        x = DenseLayer(x, n_units=8*8*hidden_number, name='Generator/dense2')
        arguments = {'shape': [-1, 8, 8, hidden_number], 'name': 'Generator/reshape1'}
        x = LambdaLayer(x, fn=tf.reshape, fn_args=arguments)
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1,1,1,1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu,name='Generator/conv1')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu,name='Generator/conv2')
        x = UpSampling2dLayer(x, size=[2, 2], is_scale=True, method=1, name='Generator/UpSampling1') # method= 1 NN

        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu,name='Generator/conv3')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Generator/conv4')
        x = UpSampling2dLayer(x, size=[2, 2], is_scale=True, method=1, name='Encoder/UpSampling2')  # method= 1 NN

        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu,name='Generator/conv5')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu,name='Generator/conv6')
        x = UpSampling2dLayer(x, size=[2, 2], is_scale=True, method=1, name='Generator/UpSampling3')  # method= 1 NN

        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1],
                        padding='SAME',
                        W_init=w_init,act=tf.nn.elu, name='Generator/conv7')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Generator/conv8')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, 3], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, name='Generator/convLAST')

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
    global_step = tf.Variable(0, name='global_step', trainable=False)
    image_width = 64
    image_height = 64
    audio_width = 12
    audio_height = 35

    # ##========================== DEFINE INPUT DATA ============================###
    images = tf.placeholder('float32', [None, image_height, image_width, 3],
                                 name='t_image_generator')
    audio = tf.placeholder('float32', [None, audio_height, audio_width, 1],
                                 name='t_audio_input_generator')
    tf.summary.image('input_image', images)
    images_normalized = norm_img(images)  # Normalization

    # ##========================== DEFINE MODEL ============================###
    net_gen = generator(input_audio=audio,reuse=False)
    tf.summary.image('norm_generated_image', net_gen.outputs)
    tf.summary.image('generated_image', denorm_img(net_gen.outputs))
    net_d, d_z = discriminator(disc_input=tf.concat([net_gen.outputs, images_normalized], axis=0), reuse=False)
    net_d_false, net_d_real = tf.split(net_d.outputs, num_or_size_splits=2, axis=0)
    d_z_false, d_z_real = tf.split(d_z.outputs, num_or_size_splits=2, axis=0)
    tf.summary.image('autoencoder_real', denorm_img(net_d_real))
    tf.summary.image('autoencoder_fake', denorm_img(net_d_false))

    output_gen = denorm_img(net_gen.outputs)  # Denormalization
    ae_gen, ae_real = denorm_img(net_d_false), denorm_img(net_d_real)  # Denormalization

    # ###========================== DEFINE TRAIN OPS ==========================###
    lambda_k = 0.001
    gamma = 0.7
    k_t = tf.Variable(0., trainable=False, name='k_t')

    g_vars = tl.layers.get_variables_with_name('generator', True, True)
    d_vars = tl.layers.get_variables_with_name('discriminator', True, True)
    with tf.variable_scope('learning_rate'):
        lr = tf.Variable(0.00008, trainable=False)

    decay_rate = 0.5
    decay_steps = 116722
    learning_rate = tf.train.inverse_time_decay(lr, decay_rate=decay_rate, decay_steps=decay_steps,
                                                global_step=global_step)

    d_loss_real = tf.reduce_mean(tf.abs(ae_real-images))
    d_loss_fake = tf.reduce_mean(tf.abs(ae_gen-output_gen))
    d_loss = d_loss_real - k_t * d_loss_fake

    g_loss_discriminativefeatures = tf.reduce_mean(tf.abs(d_z_real-d_z_false))
    g_loss = tf.reduce_mean(tf.abs(ae_gen - output_gen)) + 10e-2 * g_loss_discriminativefeatures


    g_optim = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars, global_step=global_step)
    d_optim = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars, global_step=global_step)

    balance = gamma*d_loss_real-g_loss
    with tf.control_dependencies([d_optim, g_optim]):
        k_update = tf.assign(k_t, tf.clip_by_value(k_t + lambda_k * balance, 0, 1))

    m_global = d_loss_real + tf.abs(balance)

    tf.summary.scalar('m_global', m_global)
    tf.summary.scalar('g_loss_discriminativefeatures', g_loss_discriminativefeatures)
    tf.summary.scalar('k_t', k_t)
    tf.summary.scalar('learning_rate', learning_rate)

    summary = tf.summary.merge_all()
    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=1, var_list=tf.trainable_variables())
        # Summary writer to save logs
        summary_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'), sess.graph)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        if args.resume == "True":
            print("Restoring model from checkpoint")
            restore_model(sess)

        # Coordinate the different workers for the input data pipeline
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(coord=coord)

        items_faces, items_audio = dataset.get_items()
        total = 0
        for j in range(0, epochs):
            iteration = 0
            while iteration * batch_size < len(items_faces):
                input_images = np.empty([batch_size, 64, 64, 3])
                audio_MFCC = np.empty([batch_size, 35, 12, 1])
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
                # ##========================= train SRGAN =========================###
                kt, mGlobal, summary_str = sess.run([k_update, m_global, summary], feed_dict={images: input_images,
                                                                                              audio: audio_MFCC})
                print("Epoch: %2d Iteration: %2d kt: %.8f Mglobal: %.8f." % (j, iteration, kt, mGlobal))
                summary_writer.add_summary(summary_str, total)

                # summary_writer.flush()

                # ##========================= save checkpoint =========================###
                if iteration % 3630 == 0 and iteration > 0:
                    tf.logging.info('Saving checkpoint')
                    saver.save(sess, args.checkpoint_dir + "/checkpoint", global_step=iteration, write_meta_graph=False)
                iteration += 1
                total += 1
            rest = len(items_faces) - ((iteration - 1)*batch_size)
            if rest > 0:
                count = 0
                input_images = np.empty([rest, 64, 64, 3])
                audio_MFCC = np.empty([rest, 35, 12, 1])
                for face, input_audio in zip(items_faces[len(items_faces)-rest:], items_audio[len(items_faces)-rest:]):
                    input_image = Image.open(face)
                    input_image = np.asarray(input_image, dtype=float)
                    input_images[count] = input_image
                    input_audio = np.load(input_audio)
                    input_audio = np.asarray(input_audio, dtype=float)
                    audio_MFCC[count] = input_audio[:, :, np.newaxis]
                    count += 1
                # ##========================= train SRGAN =========================###
                kt, mGlobal, summary_str = sess.run([k_update, m_global, summary], feed_dict={images: input_images,
                                                                                              audio: audio_MFCC})
                print("Iteration: %2d kt: %.8f Mglobal: %.8f." % (iteration, kt, mGlobal))
                summary_writer.add_summary(summary_str, iteration)


        # except tf.errors.OutOfRangeError:
        #     print('Done -- epoch limit reached')
        # finally:
        #     coord.request_stop()
        #     coord.join(threads)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predict script')
    parser.add_argument('-dataset_faces_folder', default=DEFAULT_DATA_FACES_PATH, help='Path to the images file')
    parser.add_argument('-dataset_audios_folder', default=DEFAULT_DATA_AUDIOS_PATH, help='Path to the audios file')
    parser.add_argument('-checkpoint_dir', default=DEFAULT_CHECKPOINT_DIR, help='Model checkpoint to use')
    parser.add_argument('-log_dir', default=DEFAULT_LOG_DIR, help='Model checkpoint to use')
    parser.add_argument('-resume', default="True", help='Resume training ("True" or "False")')

    args = parser.parse_args()

    # if args.resume == "False":
    #     if tf.gfile.Exists(args.log_dir):
    #         tf.gfile.DeleteRecursively(args.log_dir)
    #     tf.gfile.MakeDirs(args.log_dir)

    # if not os.path.isdir(os.path.dirname(args.checkpoint_dir)):
    #     os.mkdir(os.path.dirname(args.checkpoint_dir))

    train(batch_size=16, epochs=10, dataset=DataInput(args.dataset_faces_folder, args.dataset_audios_folder, "train"), log_dir=args.log_dir)
