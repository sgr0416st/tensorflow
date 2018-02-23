from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
from deep_MNIST_model import DeepNeuralNetwork

import tensorflow as tf
import argparse
import sys

FLAGS = None


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784], name='input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='answer')

    # Build the graph for the deep net
    model = DeepNeuralNetwork()
    y_conv, keep_prob = model.deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy_check'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        tf.add_to_collection('accuracy', accuracy)

        saver = tf.train.Saver()

    with tf.Session() as sess:
        # saver = tf.train.import_meta_graph('models/deep_MNIST_save/ver1.1.meta')
        saver.restore(sess, 'models/deep_MNIST_save/ver1.2')
        accuracy = tf.get_collection('accuracy')[0]

        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
