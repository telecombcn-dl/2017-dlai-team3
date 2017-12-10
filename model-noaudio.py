import tensorlayer as tl
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def restore_model(sess, checkpoint_path):
    # Get the state of the checkpoint and then restore using ckpt path
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)

    if checkpoint_path is not None:
        restorer = tf.train.Saver()
        restorer.restore(sess, ckpt.model_checkpoint_path)


def generator(z, reuse, hidden_number=64, kernel=3):
    w_init = tf.random_normal_initializer(stddev=0.02)

    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        # DECODER BEGINS
        # hidden_number = n = 128
        # exponential linear units output convolutions
        # Each layer is repeated a number of times (typically 2). We observed that more repetitions led to
        # even better visual results
        # Down-sampling is implemented as sub-sampling with stride 2 and up- sampling is done by nearest neighbor.
        x = InputLayer(z, name="in")
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

        return x


def discriminator(disc_input, reuse, z_num=64, hidden_number=64, kernel=3):
    w_init = tf.random_normal_initializer(stddev=0.02)

    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        # Encoder
        # Down-sampling is implemented as sub-sampling with stride 2

        x = InputLayer(disc_input, name='in')  # [1, height = 64, width = 64, 3 ]
        x = Conv2dLayer(x, shape=[kernel, kernel, 3, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Discriminator/Encoder/conv1')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Discriminator/Encoder/conv2')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, 2 * hidden_number], strides=[1, 1, 1, 1],
                        padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Discriminator/Encoder/conv3')
        x = Conv2dLayer(x, shape=[kernel, kernel, 2 * hidden_number, 2 * hidden_number], strides=[1, 2, 2, 1],
                        padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Discriminator/Encoder/subsampling1')
        # [1, height = 32, width = 32, 2*hidden_number]

        x = Conv2dLayer(x, shape=[kernel, kernel, 2 * hidden_number, 2 * hidden_number], strides=[1, 1, 1, 1],
                        padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Discriminator/Encoder/conv4')
        x = Conv2dLayer(x, shape=[kernel, kernel, 2 * hidden_number, 3 * hidden_number], strides=[1, 1, 1, 1],
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
                        padding='SAME', W_init=w_init, act=tf.nn.elu, name='Discriminator/Encoder/conv8')
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

    # ##========================== DEFINE INPUT DATA ============================###
    images = tf.placeholder('float32', [None, image_height, image_width, 3], name='t_image_generator')
    z = tf.placeholder('float32', [None, 64], name='t_noise_generator')
    tf.summary.image('input_image', images)
    images_normalized = norm_img(images)  # Normalization

    # ##========================== DEFINE MODEL ============================###
    net_gen = generator(z=z, reuse=False)
    tf.summary.image('norm_generated_image', net_gen.outputs)
    tf.summary.image('generated_image', denorm_img(net_gen.outputs))
    net_d, d_z = discriminator(disc_input=tf.concat([net_gen.outputs, images_normalized], axis=0), reuse=False)
    net_d_false, net_d_real = tf.split(net_d.outputs, num_or_size_splits=2, axis=0)
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
        lr = tf.Variable(0.00004, trainable=False)

    d_loss_real = tf.reduce_mean(tf.abs(ae_real - images))
    d_loss_fake = tf.reduce_mean(tf.abs(ae_gen - output_gen))
    d_loss = d_loss_real - k_t * d_loss_fake

    g_loss = tf.reduce_mean(tf.abs(ae_gen - output_gen))

    g_optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(g_loss, var_list=g_vars, global_step=global_step)
    d_optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(d_loss, var_list=d_vars, global_step=global_step)

    balance = gamma * d_loss_real - g_loss
    with tf.control_dependencies([d_optim, g_optim]):
        k_update = tf.assign(k_t, tf.clip_by_value(k_t + lambda_k * balance, 0, 1))

    m_global = d_loss_real + tf.abs(balance)

    tf.summary.scalar('m_global', m_global)
    tf.summary.scalar('g_loss', g_loss)
    tf.summary.scalar('d_loss', d_loss)
    tf.summary.scalar('k_t', k_t)

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
                # ##========================= train BEGAN =========================###
                kt, mGlobal, summary_str = sess.run([k_update, m_global, summary],
                                       feed_dict={images: input_images, z: input_z})
                summary_writer.add_summary(summary_str, total)
                if iteration % 16 == 0 and iteration > 0:
                    print("Epoch: %2d Iteration: %2d kt: %.8f Mglobal: %.8f." % (j, iteration, kt, mGlobal))

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
                # ##========================= train BEGAN =========================###
                kt, mGlobal, summary_str = sess.run([k_update, m_global, summary],
                                                    feed_dict={images: input_images, z: input_z})
                print("Iteration: %2d kt: %.8f Mglobal: %.8f." % (iteration, kt, mGlobal))
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

    train(batch_size=1, epochs=10, dataset=DataInput(args.dataset_faces_folder, args.dataset_audios_folder,
                                                      "train"), log_dir=args.log_dir)
