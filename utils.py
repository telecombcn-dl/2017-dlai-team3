import tensorflow as tf

def norm_img(image):
    image = image/127.5 - 1.
    return image

def denorm_img(norm):
    return tf.clip_by_value((norm + 1)*127.5, 0, 255)