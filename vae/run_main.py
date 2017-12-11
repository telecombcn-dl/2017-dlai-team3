import tensorflow as tf
import numpy as np
import data_input
import os
import vae
import glob

import argparse

IMAGE_SIZE_MNIST = 28

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of 'Variational AutoEncoder (VAE)'"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--results_path', type=str, default='results',
                        help='File path of output images')

    parser.add_argument('--add_noise', type=bool, default=False, help='Boolean for adding salt & pepper noise to input image')

    parser.add_argument('--dim_z', type=int, default='20', help='Dimension of latent vector', required = True)

    parser.add_argument('--n_hidden', type=int, default=500, help='Number of hidden units in MLP')

    parser.add_argument('--learn_rate', type=float, default=1e-7, help='Learning rate for Adam optimizer')

    parser.add_argument('--num_epochs', type=int, default=300, help='The number of epochs to run')

    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

    # ------- Changed PRR to False so the results are not printed
    parser.add_argument('--PRR', type=bool, default=False,
                        help='Boolean for plot-reproduce-result')

    parser.add_argument('--PRR_n_img_x', type=int, default=10,
                        help='Number of images along x-axis')

    parser.add_argument('--PRR_n_img_y', type=int, default=10,
                        help='Number of images along y-axis')

    parser.add_argument('--PRR_resize_factor', type=float, default=1.0,
                        help='Resize factor for each displayed image')

    # -------- Not printing Manifold learning result
    parser.add_argument('--PMLR', type=bool, default=False,
                        help='Boolean for plot-manifold-learning-result')

    parser.add_argument('--PMLR_n_img_x', type=int, default=20,
                        help='Number of images along x-axis')

    parser.add_argument('--PMLR_n_img_y', type=int, default=20,
                        help='Number of images along y-axis')

    parser.add_argument('--PMLR_resize_factor', type=float, default=1.0,
                        help='Resize factor for each displayed image')

    parser.add_argument('--PMLR_z_range', type=float, default=2.0,
                        help='Range for unifomly distributed latent vector')

    parser.add_argument('--PMLR_n_samples', type=int, default=5000,
                        help='Number of samples in order to get distribution of labeled data')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):

    # delete all existing files
    files = glob.glob(args.results_path+'/*')
    for f in files:
        os.remove(f)

    # --add_noise
    try:
        assert args.add_noise == True or args.add_noise == False
    except:
        print('add_noise must be boolean type')
        return None

    # --dim-z
    try:
        assert args.dim_z > 0
    except:
        print('dim_z must be positive integer')
        return None

    # --n_hidden
    try:
        assert args.n_hidden >= 1
    except:
        print('number of hidden units must be larger than one')

    # --learn_rate
    try:
        assert args.learn_rate > 0
    except:
        print('learning rate must be positive')

    # --num_epochs
    try:
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    # --PRR
    try:
        assert args.PRR == True or args.PRR == False
    except:
        print('PRR must be boolean type')
        return None

    if args.PRR == True:
        # --PRR_n_img_x, --PRR_n_img_y
        try:
            assert args.PRR_n_img_x >= 1 and args.PRR_n_img_y >= 1
        except:
            print('PRR : number of images along each axis must be larger than or equal to one')

        # --PRR_resize_factor
        try:
            assert args.PRR_resize_factor > 0
        except:
            print('PRR : resize factor for each displayed image must be positive')

    # --PMLR
    try:
        assert args.PMLR == True or args.PMLR == False
    except:
        print('PMLR must be boolean type')
        return None

    if args.PMLR == True:
        try:
            assert args.dim_z == 2
        except:
            print('PMLR : dim_z must be two')

        # --PMLR_n_img_x, --PMLR_n_img_y
        try:
            assert args.PMLR_n_img_x >= 1 and args.PMLR_n_img_y >= 1
        except:
            print('PMLR : number of images along each axis must be larger than or equal to one')

        # --PMLR_resize_factor
        try:
            assert args.PMLR_resize_factor > 0
        except:
            print('PMLR : resize factor for each displayed image must be positive')

        # --PMLR_z_range
        try:
            assert args.PMLR_z_range > 0
        except:
            print('PMLR : range for unifomly distributed latent vector must be positive')

        # --PMLR_n_samples
        try:
            assert args.PMLR_n_samples > 100
        except:
            print('PMLR : Number of samples in order to get distribution of labeled data must be large enough')

    return args

"""main function"""
def main(args):

    n_hidden = args.n_hidden
    dim_img = 35*11
    dim_z = args.dim_z

    # train
    n_epochs = args.num_epochs
    batch_size = args.batch_size
    learn_rate = args.learn_rate

    """ prepare MNIST data """
    # Change data to imput MFCC
    train_data_, train_size = data_input.prepare_MFCC_dataset()
    n_samples = train_size

    """ build graph """

    # input placeholders
    # In denoising-autoencoder, x_hat == x + noise, otherwise x_hat == x
    x_hat = tf.placeholder(tf.float32, shape=[None, dim_img], name='input_img')
    x = tf.placeholder(tf.float32, shape=[None, dim_img], name='target_img')
    tf.summary.image('target_img', tf.reshape(x, shape=[-1, 35, 11, 1]))

    # dropout
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # network architecture
    y, z, loss, neg_marginal_likelihood, KL_divergence = vae.autoencoder(x_hat, x, dim_img, dim_z, n_hidden, keep_prob)

    tf.summary.image('output_img', tf.reshape(y, shape=[-1, 35, 11, 1]))
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('neg_marginal_likelihood', neg_marginal_likelihood)
    tf.summary.scalar('KL_divergence', KL_divergence)

    # optimization
    train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)

    """ training """
    # train
    total_batch = int(n_samples / batch_size)
    summary = tf.summary.merge_all()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer(), feed_dict={keep_prob : 0.9})
        summary_writer = tf.summary.FileWriter(os.path.join("/storage/logs", 'train'), sess.graph)
        total = 0
        for epoch in range(n_epochs):

            # Random shuffling
            np.random.shuffle(train_data_)

            # Loop over all batches
            for i in range(total_batch):
                # Compute the offset of the current minibatch in the data.
                offset = (i * batch_size) % n_samples
                batch_xs_input = train_data_[offset:(offset + batch_size), :]

                batch_xs_target = batch_xs_input

                _, tot_loss, loss_likelihood, loss_divergence, summary_str = sess.run(
                    (train_op, loss, neg_marginal_likelihood, KL_divergence, summary),
                    feed_dict={x_hat: batch_xs_input, x: batch_xs_target, keep_prob : 0.9})
                summary_writer.add_summary(summary_str, total)
                total += 1

            print("epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f" % (epoch, tot_loss,
                                                                                      loss_likelihood,
                                                                                      loss_divergence))


if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    # main
    main(args)