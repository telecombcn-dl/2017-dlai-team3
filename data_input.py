import os
import tensorflow as tf


def _preprocess_example(example):
    image = example[0]

    with tf.name_scope('image_normalization'):
        # Imagenet mean values per channel in BGR format
        imagenet_mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32, name='imagenet_mean')
        image = tf.reverse(image, axis=[-1], name='rgb_to_bgr')
        image = tf.subtract(image, imagenet_mean, name='subtract_imagenet_mean')

    return image, example[1]


def _read_item(queue):
    reader = tf.WholeFileReader()
    image_path, encoded_image = reader.read(queue)

    # Read and decode image
    with tf.name_scope('decode_image'):
        image = tf.image.decode_jpeg(encoded_image, channels=3)
        image = tf.to_float(image)
        image = tf.reshape(image, [224, 224, 3])

    return image, image_path


def _create_batch(example, batch_size, num_threads):
    # Specify the samples queue parameters
    min_after_dequeue = 10 * batch_size
    capacity = min_after_dequeue + (num_threads + 3) * batch_size
    images_batch, filename = tf.train.batch(example, batch_size=batch_size, capacity=capacity,
                                            num_threads=num_threads)
    return images_batch, filename


class DataInput(object):

    def __init__(self, path, name):
        self.path = path
        self.name = name

    def input_pipeline(self, batch_size, num_epochs, shuffle=True, num_threads=1):
        with tf.device('/cpu:0'):
            with tf.name_scope(self.name):
                items = self._get_input_queue_items()
                queue = tf.train.string_input_producer(items, num_epochs=num_epochs, shuffle=shuffle)
                example = _read_item(queue)
                example = _preprocess_example(example)
                return _create_batch(example, batch_size, num_threads)

    def decode_jpeg(image_buffer, scope=None):
        """Decode a JPEG string into one 3-D float image Tensor.
        Args:
          image_buffer: scalar string Tensor.
          scope: Optional scope for op_scope.
        Returns:
          3-D float Tensor with values ranging from [0, 1).
        """
        with tf.name_scope(scope, 'decode_jpeg', [image_buffer]):
            # Decode the string as an RGB JPEG.
            # Note that the resulting image contains an unknown height and width
            # that is set dynamically by decode_jpeg. In other words, the height
            # and width of image is unknown at compile-time.
            image = tf.image.decode_jpeg(image_buffer, channels=3)

            # After this point, all image pixels reside in [0,1)
            # until the very end, when they're rescaled to (-1, 1).  The various
            # adjust_* ops all require this range for dtype float.
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            return image

    def _get_input_queue_items(self):
        """Method that should return a list of items that contain the dataset."""
        images_filenames = [os.path.join(self.path, f) for f in os.listdir(self.path)
                            if os.path.isfile(os.path.join(self.path, f))]

        images_filenames.sort()
        return images_filenames


if __name__ == '__main__':

    data_path = "/path/to/data"

    dataset = DataInput(data_path, "train")

    images, filename = dataset.input_pipeline(1, 1, shuffle=True)

    with tf.Session() as sess:
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        # Coordinate the different workers for the input data pipeline
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(50):
            im, filename_out = sess.run([images, filename])
            # Test whatever

        coord.request_stop()
        coord.join(threads)
