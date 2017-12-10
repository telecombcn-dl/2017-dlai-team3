import tensorflow as tf


def norm_img(image):
    image = image/127.5 - 1.
    return image


def denorm_img(norm):
    return tf.clip_by_value((norm + 1)*127.5, 0, 255)


def smooth_gan_labels(y):
    if y == 0:
        y_out = tf.random_uniform(shape=[-1, 1], minval=0.0, maxval=0.3)
    else:
        y_out = tf.random_uniform(shape=[-1, 1], minval=0.7, maxval=1.2)

    return y_out
