import tensorlayer as tl
import tensorflow as tf
import os
from tensorlayer.layers import *
from data_input import DataInput
from utils import norm_img, denorm_img, smooth_gan_labels


import argparse
from PIL import Image

DEFAULT_DATA_FACES_PATH = "/storage/dataset"
DEFAULT_DATA_AUDIOS_PATH = "/storage/dataset_videos/cropped_videos/outputb"
DEFAULT_LOG_DIR = "/storage/logs"
DEFAULT_CHECKPOINT_DIR = "/storage/checkpoints"


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def lrelu2(x, name="lrelu"):
    return tf.maximum(x, 0.3*x)


def restore_model(sess, checkpoint_path):
    # Get the state of the checkpoint and then restore using ckpt path
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)

    if checkpoint_path is not None:
        restorer = tf.train.Saver()
        restorer.restore(sess, ckpt.model_checkpoint_path)


# TODO: ADD SKIP CONNECTIONS (To improve performance, not in the original began paper)
def generator(z, reuse, hidden_number=64, kernel=3):
    w_init = tf.random_normal_initializer(stddev=0.02)

    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        # EXTRACT AUDIO FEATURES
        # x = InputLayer(gen_input, name="in") #[batch_size, height, width, 1]
        # x = Conv2dLayer(x, shape=[kernel, kernel, 1, 64], strides=[1, 1, 1, 1], padding='SAME', W_init=w_init,
        #                 name='AudioFeatures/conv1')
        # x = Conv2dLayer(x, shape=[kernel, kernel, 64, 128], strides=[1, 1, 1, 1], padding='SAME', W_init=w_init,
        #                 name='AudioFeatures/conv2')
        # # max o avg pool?
        # x = PoolLayer(x,strides=[1, 2, 1, 1], pool=tf.nn.avg_pool, name='AudioFeatures/pool1')
        # x = Conv2dLayer(x, shape=[kernel, kernel, 128, 256], strides=[1, 1, 1, 1], padding='SAME', W_init=w_init,
        #                 name='AudioFeatures/conv3')
        # x = Conv2dLayer(x, shape=[kernel, kernel, 256, 512], strides=[1, 1, 1, 1], padding='SAME', W_init=w_init,
        #                 name='AudioFeatures/conv4')
        # x = PoolLayer(x, strides=[1, 2, 1, 1], pool=tf.nn.avg_pool, name='AudioFeatures/pool2')
        # x = FlattenLayer(x, name='AudioFeatures/flatten')
        # x = DenseLayer(x, n_units=512, name='AudioFeatures/dense1')
        # x = DenseLayer(x, n_units=256, name='AudioFeatures/dense2') #[batch_size, 256]

        # DECODER BEGINS
        # hidden_number = n = 128
        # exponential linear units output convolutions
        # Each layer is repeated a number of times (typically 2). We observed that more repetitions led to
        # even better visual results
        # Down-sampling is implemented as sub-sampling with stride 2 and up- sampling is done by nearest neighbor.
        x = InputLayer(z, name="in")
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



def discriminator(disc_input, reuse, kernel=3, is_train=True):
    w_init = tf.random_normal_initializer(stddev=0.02)

    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        x = InputLayer(disc_input, name='in')
        x = Conv2dLayer(x, act=lrelu2, shape=[kernel, kernel, 3, 32], strides=[1, 1, 1, 1],
                        padding='SAME', W_init=w_init, name='conv1')
        x = Conv2dLayer(x, shape=[kernel, kernel, 32, 32], strides=[1, 2, 2, 1],
                        padding='SAME', W_init=w_init, name='conv2')

        x = BatchNormLayer(x, is_train=is_train, name='BN1-conv2', act=lrelu2)

        x = Conv2dLayer(x, shape=[kernel, kernel, 32, 64], strides=[1, 1, 1, 1],
                        padding='SAME', W_init=w_init, name='conv3')
        x = BatchNormLayer(x, is_train=is_train, name='BN1-conv3', act=lrelu2)
        x = Conv2dLayer(x, shape=[kernel, kernel, 64, 64], strides=[1, 2, 2, 1],
                        padding='SAME', W_init=w_init, name='conv4')
        x = BatchNormLayer(x, is_train=is_train, name='BN1-conv4', act=lrelu2)

        x = Conv2dLayer(x, shape=[kernel, kernel, 64, 128], strides=[1, 1, 1, 1],
                        padding='SAME', W_init=w_init, name='conv5')
        x = BatchNormLayer(x, is_train=is_train, name='BN1-conv5', act=lrelu2)
        x = Conv2dLayer(x, shape=[kernel, kernel, 128, 128], strides=[1, 2, 2, 1],
                        padding='SAME', W_init=w_init, name='conv6')
        x = BatchNormLayer(x, is_train=is_train, name='BN1-conv6', act=lrelu2)

        x = Conv2dLayer(x, shape=[kernel, kernel, 128, 256], strides=[1, 1, 1, 1],
                        padding='SAME', W_init=w_init, name='conv7')
        x = BatchNormLayer(x, is_train=is_train, name='BN1-conv7', act=lrelu2)
        x = Conv2dLayer(x, shape=[kernel, kernel, 256, 256], strides=[1, 2, 2, 1],
                        padding='SAME', W_init=w_init, name='conv8')
        x = BatchNormLayer(x, is_train=is_train, name='BN1-conv8', act=lrelu2)

        x = FlattenLayer(x, name='flatten')
        x = DenseLayer(x, n_units=1024, act=lrelu2, name='dense1')
        x = DenseLayer(x, n_units=1, name='dense2')

        logits = x.outputs
        x.outputs = tf.nn.sigmoid(x.outputs, name='output')

        return x, logits


def train(batch_size, epochs, dataset, log_dir):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    image_width = 64
    image_height = 64

    # ##========================== DEFINE INPUT DATA ============================###
    images = tf.placeholder('float32', [None, image_height, image_width, 3], name='t_image_generator')
    z = tf.placeholder('float32', [None, 64], name='t_noise_generator')
    y_gan_real = tf.placeholder('float32', [None, 1], name='t_labels_real')
    y_gan_fake = tf.placeholder('float32', [None, 1], name='t_labels_fake')
    y_generator = tf.placeholder('float32', [None, 1], name='t_labels_generator')
    tf.summary.image('input_image', images)
    images_normalized = norm_img(images)  # Normalization

    # ##========================== DEFINE MODEL ============================###
    net_gen = generator(z=z, reuse=False)
    tf.summary.image('generated_normalized_image', net_gen.outputs)
    tf.summary.image('generated_image', denorm_img(net_gen.outputs))
    net_d, logits = discriminator(disc_input=tf.concat([net_gen.outputs, images_normalized], axis=0), reuse=False)
    net_d_false, net_d_real = tf.split(net_d.outputs, num_or_size_splits=2, axis=0)

    # ###========================== DEFINE TRAIN OPS ==========================###

    g_vars = tl.layers.get_variables_with_name('generator', True, True)
    d_vars = tl.layers.get_variables_with_name('discriminator', True, True)
    with tf.variable_scope('learning_rate'):
        lr = tf.Variable(1e-4, trainable=False)

    d_loss_real = tf.reduce_mean(tf.square(net_d_real - y_gan_real), name='d_loss_real')
    d_loss_fake = tf.reduce_mean(tf.square(net_d_false - y_gan_fake), name='d_loss_fake')
    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(tf.square(net_d_false - y_generator), name='g_loss_gan')
    g_optim = tf.train.AdamOptimizer(lr).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr).minimize(d_loss, var_list=d_vars)


    tf.summary.scalar('d_loss', d_loss)
    tf.summary.scalar('g_loss', g_loss)

    summary = tf.summary.merge_all()
    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=1)
        # Summary writer to save logs
        summary_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'), sess.graph)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        if args.resume == "True":
            print("Restoring model from checkpoint")
            restore_model(sess, args.checkpoint_dir)

        items_faces, items_audio = dataset.get_items()
        total = 0
        for j in range(0, epochs):
            iteration = 0
            while iteration * batch_size < len(items_faces):
                input_images = np.empty([batch_size, 64, 64, 3])
                count = 0
                for face in items_faces[iteration * batch_size:iteration * batch_size + batch_size]:
                    input_image = Image.open(face)
                    input_image = np.asarray(input_image, dtype=float)
                    input_images[count] = input_image
                    count += 1
                input_z = np.random.uniform(-1., 1, size=[batch_size, 64])

                if np.random.uniform() > 0.1:
                    # give correct classifications
                    labels_real = np.random.uniform(size=[batch_size, 1], low=0.7, high=1.2)
                    labels_fake = np.random.uniform(size=[batch_size, 1], low=0.0, high=0.3)
                else:
                    # give wrong classifications (noisy labels)
                    labels_fake = np.random.uniform(size=[batch_size, 1], low=0.7, high=1.2)
                    labels_real = np.random.uniform(size=[batch_size, 1], low=0.0, high=0.3)

                labels_generator = np.random.uniform(size=[batch_size, 1], low=0.7, high=1.2)

                # ##========================= train LSGAN =========================###
                summary_str, gLoss, dLoss, _, _ = sess.run([summary, g_loss, d_loss, g_optim, d_optim],
                                              feed_dict={images: input_images, z: input_z,
                                                         y_gan_real: labels_real, y_gan_fake: labels_fake,
                                                         y_generator: labels_generator})
                summary_writer.add_summary(summary_str, total)
                print("Epoch: %2d Iteration: %2d gLoss: %.8f dLoss: %.8f." % (j, iteration, gLoss, dLoss))

                # ##========================= save checkpoint =========================###
                if iteration % 3000 == 0 and iteration > 0:
                    tf.logging.info('Saving checkpoint')
                    saver.save(sess, args.checkpoint_dir + "/checkpoint", global_step=iteration, write_meta_graph=False)
                iteration += 1
                total += 1
            rest = len(items_faces) - ((iteration - 1) * batch_size)
            if rest > 0:
                count = 0
                input_images = np.empty([rest, 64, 64, 3])
                for face in items_faces[len(items_faces) - rest:]:
                    input_image = Image.open(face)
                    input_image = np.asarray(input_image, dtype=float)
                    input_images[count] = input_image
                    count += 1
                input_z = np.random.uniform(-1., 1, size=[rest, 64])
                if np.random.uniform() > 0.1:
                    # give correct classifications
                    labels_real = np.random.uniform(size=[rest, 1], low=0.7, high=1.2)
                    labels_fake = np.random.uniform(size=[rest, 1], low=0.0, high=0.3)
                else:
                    # give wrong classifications (noisy labels)
                    labels_fake = np.random.uniform(size=[rest, 1], low=0.7, high=1.2)
                    labels_real = np.random.uniform(size=[rest, 1], low=0.0, high=0.3)

                labels_generator = np.random.uniform(size=[rest, 1], low=0.7, high=1.2)

                # ##========================= train LSGAN =========================###
                summary_str, gLoss, dLoss, _, _ = sess.run([summary, g_loss, d_loss, g_optim, d_optim],
                                              feed_dict={images: input_images, z: input_z,
                                                         y_gan_real: labels_real, y_gan_fake: labels_fake,
                                                         y_generator: labels_generator})
                print("Epoch: %2d Iteration: %2d gLoss: %.8f dLoss: %.8f." % (j, iteration, gLoss, dLoss))
                summary_writer.add_summary(summary_str, iteration)



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

    train(batch_size=16, epochs=10, dataset=DataInput(args.dataset_faces_folder, args.dataset_audios_folder,
                                                     "train"), log_dir=args.log_dir)
t