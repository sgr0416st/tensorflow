from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

import argparse
import sys

FLAGS = None


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784], name='input')

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10], name='answer')

    keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')

    with tf.Session() as sess:
        # saver = tf.train.Saver()
        saver = tf.train.import_meta_graph('model.ckpt.meta')
        saver.restore(sess, 'model.ckpt')
        accuracy = tf.get_collection('accuracy')[0]

        # y_conv = tf.get_collection('y_conv')[0]
        # keep_prob = tf.get_collection('keep_prob')[0]

        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
