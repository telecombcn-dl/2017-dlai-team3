import tensorlayer as tl
import os
from tensorlayer.layers import *
from data_input import DataInput
from utils import norm_img, denorm_img
from scipy.ndimage.filters import gaussian_filter
import argparse
from PIL import Image

DEFAULT_DATA_FACES_PATH = "/storage/dataset"
DEFAULT_DATA_FACES_BLURRY_PATH = "/storage/MSE_output"
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


def generator(gen_in, reuse, hidden_number=64, kernel=3):
    w_init = tf.random_normal_initializer(stddev=0.02)

    with tf.variable_scope("generator", reuse=reuse):
        tl.layers.set_name_reuse(reuse)

        # DECODER BEGINS
        # hidden_number = n = 128
        # exponential linear units output convolutions
        # Each layer is repeated a number of times (typically 2). We observed that more repetitions led to
        # even better visual results
        # Down-sampling is implemented as sub-sampling with stride 2 and up- sampling is done by nearest neighbor.
        x = InputLayer(gen_in, name="in")

        x = Conv2dLayer(x, shape=[kernel, kernel, 3, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Generator/conv0')
        input_1_skip = x
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Generator/conv1')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Generator/conv2')
        x = ElementwiseLayer([x, input_1_skip], tf.add, name='Generator/skip1')

        input_2_skip = x
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Generator/conv3')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Generator/conv4')
        x = ElementwiseLayer([x, input_2_skip], tf.add, name='Generator/skip2')

        input_3_skip = x
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Generator/conv5')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Generator/conv6')
        x = ElementwiseLayer([x, input_3_skip], tf.add, name='Generator/skip3')

        input_4_skip = x
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1],
                        padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Generator/conv7')
        x = Conv2dLayer(x, shape=[kernel, kernel, hidden_number, hidden_number], strides=[1, 1, 1, 1], padding='SAME',
                        W_init=w_init, act=tf.nn.elu, name='Generator/conv8')
        x = ElementwiseLayer([x, input_4_skip], tf.add, name='Generator/skip4')

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


def test(batch_size, epochs, dataset, log_dir):
    # ##========================== DEFINE INPUT DATA ============================###
    generator_input = tf.placeholder('float32', [None, 64, 64, 3], name='t_input_generator')

    # ##========================== DEFINE MODEL ============================###
    net_gen = generator(gen_in=generator_input, reuse=False)

    output_gen = denorm_img(net_gen.outputs)  # Denormalization

    with tf.Session() as sess:
        if args.resume == "True":
            print("Restoring model from checkpoint")
            restore_model(sess, args.checkpoint_dir)

        _, items_faces_blurry = dataset.get_items_blurry()

        for i, face_blurry in enumerate(items_faces_blurry[0:100]):
            input_image = Image.open(face_blurry)
            input_image = np.asarray(input_image, dtype=float)
            input_blurry_0 = gaussian_filter(input_image[:, :, 0], sigma=2)
            input_blurry_1 = gaussian_filter(input_image[:, :, 1], sigma=2)
            input_blurry_2 = gaussian_filter(input_image[:, :, 2], sigma=2)
            input_images_blurry = np.empty([1, 64, 64, 3])
            input_images_blurry[:, :, :, 0] = input_blurry_0
            input_images_blurry[:, :, :, 1] = input_blurry_1
            input_images_blurry[:, :, :, 2] = input_blurry_2
            input_images_blurry = norm_img(input_images_blurry)

            output_image = sess.run(output_gen, feed_dict={generator_input: input_images_blurry})[0]

            ima = Image.fromarray(output_image.astype(np.uint8), 'RGB')
            ima.save("test_image_{}.png".format(i))
            ima = Image.fromarray(input_image.astype(np.uint8), 'RGB')
            ima.save("input_image_{}.png".format(i))


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

    test(batch_size=1, epochs=10, dataset=DataInput(args.dataset_faces_folder, args.dataset_audios_folder,
                                                      "train"), log_dir=args.log_dir)
