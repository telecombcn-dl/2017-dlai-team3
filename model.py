import tensorlayer as tl
import tensorflow as tf
from tensorlayer.layers import *
from data_input import DataInput
from utils import norm_img, denorm_img


# TODO: ADD SKIP CONNECTIONS (To improve performance, not in the original began paper)
def generator(gen_input, reuse, batch_size, hidden_number=64, kernel=3):
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
        # stride only time axis (ESTA BIEN?)
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


# z_num = 256 (Dimension Audio Features )
def discriminator(disc_input, reuse, batch_size, z_num=256, hidden_number=64, kernel=3):
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
        arguments = {'shape': [2*batch_size, 8, 8, hidden_number], 'name': 'Generator/reshape1'}
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


def train(batch_size, epochs, dataset):

    # ##========================== DEFINE PIPELINE ============================###
    images, audio = dataset.input_pipeline(batch_size=batch_size, num_epochs=epochs)
    images = norm_img(images)  # Normalization

    # ##========================== DEFINE MODEL ============================###
    net_gen = generator(gen_input=audio, batch_size=batch_size, reuse=False)
    net_d, d_z = discriminator(disc_input=tf.concat([net_gen.outputs, images], 0), batch_size=batch_size, reuse=False)
    net_d_false, net_d_real = tf.split(net_d.outputs, 2)

    output_gen = denorm_img(net_gen.outputs)  # Denormalization
    ae_gen, ae_real = denorm_img(net_d_false), denorm_img(net_d_real)  # Denormalization

    # ###========================== DEFINE TRAIN OPS ==========================###
    lambda_k = 0.001
    gamma = 0.5
    k_t = tf.Variable(0., trainable=False, name='k_t')

    g_vars = tl.layers.get_variables_with_name('generator', True, True)
    d_vars = tl.layers.get_variables_with_name('discriminator', True, True)
    with tf.variable_scope('learning_rate'):
        lr_v_g = tf.Variable(1e-4, trainable=False)
        lr_v_d = tf.Variable(1e-4, trainable=False)

    d_loss_real = tf.reduce_mean(tf.abs(ae_real-images))
    d_loss_fake = tf.reduce_mean(tf.abs(ae_gen-output_gen))

    d_loss = d_loss_real - k_t * d_loss_fake
    g_loss = tf.reduce_mean(tf.abs(ae_gen - output_gen))
    g_optim = tf.train.AdamOptimizer(lr_v_g).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v_d).minimize(d_loss, var_list=d_vars)

    balance = gamma*d_loss_real-g_loss
    with tf.control_dependencies([d_optim, g_optim]):
        k_update = tf.assign(k_t, tf.clip_by_value(k_t + lambda_k * balance, 0, 1))

    m_global = d_loss_real + tf.abs(balance)

    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        # Coordinate the different workers for the input data pipeline
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)


        try:
            while not coord.should_stop():
                # ##========================= train SRGAN =========================###
                kt, mGlobal = sess.run([k_update, m_global])
                print("kt: %.8f Mglobal: %.8f" % (kt, mGlobal))

                # ##========================= evaluate data =========================###

        except tf.errors.OutOfRangeError:
            print('Done -- epoch limit reached')
        finally:
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    data_path = "/storage/dataset_videos/audio2faces_dataset/"
    train(batch_size=64, epochs=1, dataset=DataInput(data_path, "train"))
