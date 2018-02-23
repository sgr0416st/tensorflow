# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import shutil

from tensorflow.examples.tutorials.mnist import input_data
from deep_MNIST_model import DeepNeuralNetwork

import tensorflow as tf

FLAGS = None


def main(_):
    # Import data
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
        sess.run(tf.global_variables_initializer())
        for i in range(300):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print('test accuracy %g' % sess.run(accuracy, feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

        save_path = "models/deep_MNIST_save/ver1.2"
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.mkdir(save_path)
        saver.save(sess, save_path)
        print("Model saved in file: %s" % save_path)

    with tf.Session() as new_sess:
        # new_saver = tf.train.import_meta_graph('models/deep_MNIST_save/ver1.1.meta')
        print("Model load in file: %s" % save_path)

        new_saver = tf.train.Saver()
        new_saver.restore(new_sess, save_path)
        print("Model is restored: %s" % save_path)
        new_accuracy = tf.get_collection('accuracy')[0]
        print('test new_accuracy %g' % new_accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
