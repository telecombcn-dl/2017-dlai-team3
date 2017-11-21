import tensorlayer as tl
import tensorflow as tf
import os
from tensorlayer.layers import *
from data_input import DataInput
from utils import norm_img, denorm_img, smooth_gan_labels

def lrelu2(x, name="lrelu"):
    return tf.maximum(x, 0.3*x)

def lrelu1(x, name="lrelu1"):
    return tf.maximum(x, 0.25*x)

# TODO: ADD SKIP CONNECTIONS (To improve performance, not in the original began paper)
def generator(gen_input, reuse, batch_size, nb = 6, hidden_number=64, kernel=3, is_train = True):
    w_init = tf.random_normal_initializer(stddev=0.02)

    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        # EXTRACT AUDIO FEATURES
        x = InputLayer(gen_input, name="in") #[batch_size, height, width, 1]
        x = Conv2dLayer(x, shape=[kernel, kernel, 1, 64], strides=[1, 1, 1, 1], padding='SAME', W_init=w_init,
                        name='AudioFeatures/conv1')
        x = Conv2dLayer(x, shape=[kernel, kernel, 64, 128], strides=[1, 1, 1, 1], padding='SAME', W_init=w_init,
                        name='AudioFeatures/conv2')
        # max o avg pool?
        x = PoolLayer(x,strides=[1, 2, 1, 1], pool=tf.nn.avg_pool, name='AudioFeatures/pool1')
        x = Conv2dLayer(x, shape=[kernel, kernel, 128, 256], strides=[1, 1, 1, 1], padding='SAME', W_init=w_init,
                        name='AudioFeatures/conv3')
        x = Conv2dLayer(x, shape=[kernel, kernel, 256, 512], strides=[1, 1, 1, 1], padding='SAME', W_init=w_init,
                        name='AudioFeatures/conv4')
        x = PoolLayer(x, strides=[1, 2, 1, 1], pool=tf.nn.avg_pool, name='AudioFeatures/pool2')
        x = FlattenLayer(x, name='AudioFeatures/flatten')
        x = DenseLayer(x, n_units=512, name='AudioFeatures/dense1')
        x = DenseLayer(x, n_units=256, name='AudioFeatures/dense2') #[batch_size, 256]

        # DECODER BEGINS
        # hidden_number = n = 128
        # exponential linear units output convolutions
        # Each layer is repeated a number of times (typically 2). We observed that more repetitions led to
        # even better visual results
        # Down-sampling is implemented as sub-sampling with stride 2 and up- sampling is done by nearest neighbor.
        x = DenseLayer(x, n_units=8*8*hidden_number, name='Generator/dense2')
        arguments = {'shape': [batch_size, 8, 8, hidden_number], 'name': 'Generator/reshape1'}
        x = LambdaLayer(x, fn=tf.reshape, fn_args=arguments)
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, 32], strides=[1, 1, 1, 1],
                        padding='SAME', W_init=w_init, name='conv1')
        x = BatchNormLayer(x, act=lrelu1, is_train=is_train, name='BN-conv1')
        inputRB = x
        inputadd = x

        # residual blocks
        for i in range(nb):
            x = Conv2dLayer(x, shape=[kernel, kernel, 32, 32], strides=[1, 1, 1, 1],
                            padding='SAME', W_init=w_init, name='conv1-rb/%s' % i)
            x = BatchNormLayer(x, act=lrelu1, is_train=is_train, name='BN1-rb/%s' % i)
            x = Conv2dLayer(x, shape=[kernel, kernel, 32, 32], strides=[1, 1, 1, 1],
                            padding='SAME', W_init=w_init, name='conv2-rb/%s' % i)
            x = BatchNormLayer(x, is_train=is_train, name='BN2-rb/%s' % i, )
            # short skip connection
            x = ElementwiseLayer([x, inputadd], tf.add, name='add-rb/%s' % i)
            inputadd = x

        # large skip connection
        x = Conv2dLayer(x, shape=[kernel, kernel, 32, 32], strides=[1, 1, 1, 1],
                        padding='SAME', W_init=w_init, name='conv2')
        x = BatchNormLayer(x, is_train=is_train, name='BN-conv2')
        x = ElementwiseLayer([x, inputRB], tf.add, name='add-conv2')

        # at that point, x=[batchsize,32,32,23,32]

        # upscaling block 1
        x = Conv2dLayer(x, shape=[kernel, kernel, 32, 64], act=lrelu1, strides=[1, 1, 1, 1],
                        padding='SAME', W_init=w_init, name='conv1-ub/1')
        x = UpSampling2dLayer(x, size=[2, 2], is_scale=True, method=1, name='Generator/UpSampling1')
        x = Conv2dLayer(InputLayer(x, name='in ub1 conv2'), shape=[kernel, kernel, 64, 64], act=lrelu1,
                        strides=[1, 1, 1, 1],
                        padding='SAME', W_init=w_init, name='conv2-ub/1')
        # upscaling block 2
        x = Conv2dLayer(x, shape=[kernel, kernel, 32, 64], act=lrelu1, strides=[1, 1, 1, 1],
                        padding='SAME', W_init=w_init, name='conv1-ub/2')
        x = UpSampling2dLayer(x, size=[2, 2], is_scale=True, method=1, name='Generator/UpSampling2')
        x = Conv2dLayer(InputLayer(x, name='in ub2 conv2'), shape=[kernel, kernel, 64, 64], act=lrelu1,
                        strides=[1, 1, 1, 1],
                        padding='SAME', W_init=w_init, name='conv2-ub/2')
        # upscaling block 3
        x = Conv2dLayer(x, shape=[kernel, kernel, 32, 64], act=lrelu1, strides=[1, 1, 1, 1],
                        padding='SAME', W_init=w_init, name='conv1-ub/3')
        x = UpSampling2dLayer(x, size=[2, 2], is_scale=True, method=1, name='Generator/UpSampling2')
        x = Conv2dLayer(InputLayer(x, name='in ub3 conv3'), shape=[kernel, kernel, 64, 64], act=lrelu1,
                        strides=[1, 1, 1, 1],
                        padding='SAME', W_init=w_init, name='conv2-ub/3')


        x = Conv3dLayer(x, shape=[kernel, kernel, 64, 1], strides=[1, 1, 1, 1],
                        act=tf.nn.tanh, padding='SAME', W_init=w_init, name='convlast')

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

    # ##========================== DEFINE PIPELINE ============================###
    images, audio = dataset.input_pipeline(batch_size=batch_size, num_epochs=epochs)
    tf.summary.image('input_image', images)
    tf.summary.image('audio_images', audio)
    images_normalized = norm_img(images)  # Normalization


    # ##========================== DEFINE MODEL ============================###
    net_gen = generator(gen_input=audio, batch_size=batch_size, reuse=False)
    tf.summary.image('generated_image', denorm_img(net_gen.outputs))
    net_d, logits = discriminator(disc_input=tf.concat([net_gen.outputs, images_normalized], axis=0), reuse=False)
    net_d_false, net_d_real = tf.split(net_d.outputs, num_or_size_splits=2, axis=0)

    # ###========================== DEFINE TRAIN OPS ==========================###

    g_vars = tl.layers.get_variables_with_name('generator', True, True)
    d_vars = tl.layers.get_variables_with_name('discriminator', True, True)
    with tf.variable_scope('learning_rate'):
        lr = tf.Variable(1e-4, trainable=False)
    global_step = tf.Variable(0, trainable=False)
    decay_rate = 0.5
    decay_steps = 116722*10 ^ 2  # aprox 583.6K steps each epoch
    learning_rate = tf.train.inverse_time_decay(lr, global_step=global_step, decay_rate=decay_rate,
                                                decay_steps=decay_steps)

    if np.random.uniform() > 0.1:
        # give correct classifications
        y_gan_real = tf.ones_like(net_d_real)
        y_gan_fake= tf.zeros_like(net_d_false)
    else:
        # give wrong classifications (noisy labels)
        y_gan_real = tf.zeros_like(net_d_real)
        y_gan_fake = tf.ones_like(net_d_false)

    d_loss_real = tf.reduce_mean(tf.square(net_d_real - smooth_gan_labels(y_gan_real)),
                                 name='d_loss_real')
    d_loss_fake = tf.reduce_mean(tf.square(net_d_false - smooth_gan_labels(y_gan_fake)),
                                 name='d_loss_fake')
    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(tf.square(net_d_false - smooth_gan_labels(tf.ones_like(net_d_false))),
                                        name='g_loss_gan')
    g_optim = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)


    tf.summary.scalar('d_loss', d_loss)
    tf.summary.scalar('g_loss', g_loss)
    tf.summary.scalar('learning rate', learning_rate)

    summary = tf.summary.merge_all()
    config = tf.ConfigProto(device_count={'GPU': 1})

    with tf.Session(config) as sess:
        # Summary writer to save logs
        summary_writer = tf.summary.FileWriter(os.path.join(log_dir, 'train'), sess.graph)

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        # Coordinate the different workers for the input data pipeline
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)


        try:
            iteration = 0
            while not coord.should_stop():
                iteration += 1
                # ##========================= train SRGAN =========================###
                gLoss, dLoss, _, _ = sess.run([g_loss, d_loss, g_optim, d_optim])
                print("iteration: [%2d] g_loss: %.8f d_loss: %.8f" % (iteration, gLoss, dLoss))
                summary_str = sess.run(summary)
                summary_writer.add_summary(summary_str, iteration)
                summary_writer.flush()


                # ##========================= evaluate data =========================###

        except tf.errors.OutOfRangeError:
            print('Done -- epoch limit reached')
        finally:
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    data_path = "/storage/dataset_videos/audio2faces_dataset/"
    log_dir = "/storage/irina/logs"
    train(batch_size=16, epochs=1000, dataset=DataInput(data_path, "train"), log_dir=log_dir)
