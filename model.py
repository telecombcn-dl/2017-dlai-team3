import tensorlayer as tl
import tensorflow as tf
from tensorlayer.layers import *
from data_input import DataInput

# ADD SKIP CONNECTIONS (To improve performance, not in the original began paper)
def generator(input, reuse, hidden_number=128, kernel=3):
    w_init = tf.random_normal_initializer(stddev=0.02)

    with tf.variable.scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        # EXTRACT AUDIO FEATURES
        x = InputLayer(input, name="in") #[batch_size, height, width, 1]
        x = Conv2dLayer(x, shape=[kernel, kernel, 1, 64], strides=[1, 1, 1, 1], padding='SAME', W_init=w_init,
                        name='AudioFeatures/conv1')
        x = Conv2dLayer(x, shape=[kernel, kernel, 64, 128], strides=[1, 1, 1, 1], padding='SAME', W_init=w_init,
                        name='AudioFeatures/conv2')
        #max o avg pool?
        # stride only time axis (ESTA BIEN?)
        x = PoolLayer(x,strides=[1,1,2,1],pool=tf.nn.avg_pool, name='AudioFeatures/pool1')
        x = Conv2dLayer(x, shape=[kernel, kernel, 128, 256], strides=[1, 1, 1, 1], padding='SAME', W_init=w_init,
                        name='AudioFeatures/conv3')
        x = Conv2dLayer(x, shape=[kernel, kernel, 256, 512], strides=[1, 1, 1, 1], padding='SAME', W_init=w_init,
                        name='AudioFeatures/conv4')
        x = PoolLayer(x, strides=[1, 1, 2, 1], pool=tf.nn.avg_pool, name='AudioFeatures/pool2')
        x = DenseLayer(x, n_units=512, name='AudioFeatures/dense1')
        x = DenseLayer(x, n_units=256, name='AudioFeatures/dense2') #[batch_size, 256]

        # DECODER BEGINS
        # hidden_number = n = 128
        # exponential linear units output convolutions
        # Each layer is repeated a number of times (typically 2). We observed that more repetitions led to
        # even better visual results
        # Down-sampling is implemented as sub-sampling with stride 2 and up- sampling is done by nearest neighbor.
        x = DenseLayer(x, n_units=8*8*hidden_number, name='Generator/dense2')
        x = tf.reshape(x, shape=[8,8,hidden_number], name='Generator/reshape1')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1,1,1,1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu,name='Generator/conv1')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu,name='Generator/conv2')
        x = UpSampling2dLayer(x, size=2, is_scale=True, method=1, name='Generator/UpSampling1') # method= 1 NN

        x = Conv2dLayer(x, shape=[kernel, kernel, 2*hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu,name='Generator/conv3')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init,act=tf.nn.elu, name='Generator/conv4')
        x = UpSampling2dLayer(x, size=2, is_scale=True, method=1, name='Encoder/UpSampling2')  # method= 1 NN

        x = Conv2dLayer(x, shape=[kernel, kernel, 2*hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu,name='Generator/conv5')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu,name='Generator/conv6')
        x = UpSampling2dLayer(x, size=2, is_scale=True, method=1, name='Generator/UpSampling3')  # method= 1 NN

        x = Conv2dLayer(x, shape=[kernel, kernel, 2 * hidden_number, hidden_number], strides=[1, 1, 1, 1],
                        padding='SAME',
                        W_init=w_init,act=tf.nn.elu, name='Generator/conv7')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu,name='Generator/conv8')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, 3], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=None,name='Generator/convLAST')

        return x

#z_num = 256 (Dimension Audio Features )
def discriminator(input, reuse,z_num = 256, hidden_number = 128, kernel=3):
    w_init = tf.random_normal_initializer(stddev=0.02)

    with tf.variable_scope("discriminator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        # Encoder
        #Down-sampling is implemented as sub-sampling with stride 2

        x = InputLayer(input, name='in') #[1, height = 64, width = 64, 3 ]
        x = Conv2dLayer(x, shape=[kernel, kernel, 3, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init,act=tf.nn.elu, name='Discriminator/Encoder/conv1')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init,act=tf.nn.elu, name='Discriminator/Encoder/conv2')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, 2*hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init,act=tf.nn.elu, name='Discriminator/Encoder/conv3')
        x = Conv2dLayer(x, shape=[kernel, kernel, 2*hidden_number, 2*hidden_number], strides=[1, 2, 2, 1], padding='SAME',
                        W_init=w_init,act=tf.nn.elu, name='Discriminator/Encoder/subsampling1')
        #[1, height = 32, width = 32, 2*hidden_number]

        x = Conv2dLayer(x, shape=[kernel, kernel, 2*hidden_number, 2*hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init,act=tf.nn.elu, name='Discriminator/Encoder/conv4')
        x = Conv2dLayer(x, shape=[kernel, kernel, 2*hidden_number, 3 * hidden_number], strides=[1, 1, 1, 1],
                        padding='SAME', W_init=w_init, act=tf.nn.elu, name='Discriminator/Encoder/conv5')
        x = Conv2dLayer(x, shape=[kernel, kernel, 3 * hidden_number, 3 * hidden_number], strides=[1, 2, 2, 1],
                        padding='SAME', W_init=w_init, act=tf.nn.elu, name='Discriminator/Encoder/subsampling2')
        #[1, height = 16, width = 16, 3*hidden_number]

        x = Conv2dLayer(x, shape=[kernel, kernel, 3 * hidden_number, 3 * hidden_number], strides=[1, 1, 1, 1],
                        padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Discriminator/Encoder/conv6')
        x = Conv2dLayer(x, shape=[kernel, kernel, 3 * hidden_number, 4 * hidden_number], strides=[1, 1, 1, 1],
                        padding='SAME', W_init=w_init, act=tf.nn.elu, name='Discriminator/Encoder/conv7')
        x = Conv2dLayer(x, shape=[kernel, kernel, 4 * hidden_number, 4 * hidden_number], strides=[1, 2, 2, 1],
                        padding='SAME', W_init=w_init, act=tf.nn.elu, name='Discriminator/Encoder/subsampling3')
        # [1, height = 8, width = 8, 4*hidden_number]

        x = Conv2dLayer(x, shape=[kernel, kernel, 4 * hidden_number, 4 * hidden_number], strides=[1, 1, 1, 1],
                        padding='SAME',  W_init=w_init, act=tf.nn.elu, name='Discriminator/Encoder/conv6')
        x = Conv2dLayer(x, shape=[kernel, kernel, 4 * hidden_number, 4 * hidden_number], strides=[1, 1, 1, 1],
                        padding='SAME', W_init=w_init, act=tf.nn.elu, name='Discriminator/Encoder/conv7')

        z = DenseLayer(x, n_units=z_num, name = 'Discriminator/Encoder/Dense')

        # Decoder
        x = DenseLayer(z, n_units=8 * 8 * hidden_number, name='Discriminator/Decoder/dense2')
        x = tf.reshape(x, shape=[8, 8, hidden_number], name='Discriminator/Decoder/reshape1')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Discriminator/Decoder/conv1')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Discriminator/Decoder/conv2')
        x = UpSampling2dLayer(x, size=2, is_scale=True, method=1, name='Discriminator/Decoder/UpSampling1')  # method= 1 NN

        x = Conv2dLayer(x, shape=[kernel, kernel, 2 * hidden_number, hidden_number], strides=[1, 1, 1, 1],
                        padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Discriminator/Decoder/conv3')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Discriminator/Decoder/conv4')
        x = UpSampling2dLayer(x, size=2, is_scale=True, method=1, name='Discriminator/Decoder/UpSampling2')  # method= 1 NN

        x = Conv2dLayer(x, shape=[kernel, kernel, 2 * hidden_number, hidden_number], strides=[1, 1, 1, 1],
                        padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Discriminator/Decoder/conv5')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Discriminator/Decoder/conv6')
        x = UpSampling2dLayer(x, size=2, is_scale=True, method=1, name='Discriminator/Decoder/UpSampling3')  # method= 1 NN

        x = Conv2dLayer(x, shape=[kernel, kernel, 2 * hidden_number, hidden_number], strides=[1, 1, 1, 1],
                        padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Discriminator/Decoder/conv7')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Discriminator/Decoder/conv8')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, 3], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=None, name='Discriminator/Decoder/convLAST')

        return z, x

#audio_width/height
#image_width/height
def train(batch_size, epochs, dataset, num_gpus):
    g_loss = []
    d_loss = []
    k_t = []
    with tf.variable_scope(tf.get_variable_scope()):
        for i in xrange(num_gpus):
            with tf.device('/gpu:{gpu_id}'.format(gpu_id=i)):
                with tf.name_scope('tower_{}'.format(i)) as scope:
                    ###========================== DEFINE PIPELINE ============================###
                    real_images, audio, filename = dataset.input_pipeline(batch_size=batch_size, num_epochs=epochs, shuffle=True)


                    ###========================== DEFINE MODEL ============================###
                    net_gen = generator(input=audio, reuse=False)
                    net_d_real, d_z_false = discriminator(input=real_images, reuse=False)
                    net_d_false, d_z_false = discriminator(input=net_gen.outputs, reuse=True)

                    # ###========================== DEFINE TRAIN OPS ==========================###

                    lambda_k = 0.001
                    gamma = 0.5
                    k_t = tf.Variable(0., trainable=False, name='k_t')

                    g_vars = tl.layers.get_variables_with_name('generator', True, True)
                    d_vars = tl.layers.get_variables_with_name('discriminator', True, True)
                    with tf.variable_scope('learning_rate'):
                        lr_v_g = tf.Variable(1e-4, trainable=False)
                        lr_v_d = tf.Variable(1e-4, trainable=False)

                    d_loss_real = tf.reduce_mean(tf.abs(net_d_real.outputs-real_images))
                    d_loss_fake = tf.reduce_mean(tf.abs(net_d_false.outputs-net_gen.outputs))

                    d_loss = d_loss_real - k_t * d_loss_fake
                    g_loss = tf.reduce_mean(tf.abs(net_d_false-net_gen.outputs))
                    g_gradients = tf.train.AdamOptimizer(lr_v_g).compute_gradients(g_loss, var_list=g_vars)
                    d_gradients = tf.train.AdamOptimizer(lr_v_d).compute_gradients(d_loss, var_list=d_vars)

                    balance = gamma*d_loss_real-g_loss
                    with tf.control_dependencies([d_optim, g_optim]):
                        k_update = tf.assign(k_t, tf.clip_by_value(k_t + lambda_k * balance , 0, 1))

                    m_global = d_loss_real + tf.abs(balance)


    session = tf.Session()
    tl.layers.initialize_global_variables(session)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for j in range(0, epochs):
        ###========================= train GAN =========================###
        # update D
        errD, _ = session.run([d_loss, d_optim])
        # update G
        errG,mGlobal, _ = session.run([g_loss,m_global, g_optim])
        print("Epoch [%2d/%2d] : d_loss: %.8f g_loss: %.8f m_global: %.8f " % (j, epochs,  errD, errG, mGlobal))

        ###========================= evaluate data =========================###


if __name__ == '__main__':

    data_path = "/path/to/data"
    train(batch_size=64, epochs=10, dataset=DataInput(data_path, "train"), num_gpus=1)
