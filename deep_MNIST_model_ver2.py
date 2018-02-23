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

import numpy as np
import tensorflow as tf

FLAGS = None


class DeepNeuralNetwork:

    def __init__(self, sess):
        self.x = tf.placeholder(tf.float32, [None, 784], name='input')
        self.y_ = tf.placeholder(tf.float32, [None, 10], name='answer')
        with tf.name_scope('reshape'):
            self.x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        # First convolutional layer - maps one grayscale image to 32 feature maps.
        with tf.name_scope('conv1'):
            self.W_conv1 = self.weight_variable([5, 5, 1, 32], "W_conv1")
            self.b_conv1 = self.bias_variable([32], "b_conv1")
            self.h_conv1 = tf.nn.relu(tf.nn.conv2d(self.x_image, self.W_conv1, strides=[1, 1, 1, 1], padding='SAME')
                                      + self.b_conv1)
        # Pooling layer - down samples by 2X.
        with tf.name_scope('pool1'):
            self.h_pool1 = tf.nn.max_pool(self.h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # Second convolutional layer -- maps 32 feature maps to 64.
        with tf.name_scope('conv2'):
            self.W_conv2 = self.weight_variable([5, 5, 32, 64], "W_conv2")
            self.b_conv2 = self.bias_variable([64], "b_conv2")
            self.h_conv2 = tf.nn.relu(tf.nn.conv2d(self.h_pool1, self.W_conv2, strides=[1, 1, 1, 1], padding='SAME')
                                      + self.b_conv2)
        # Second pooling layer.
        with tf.name_scope('pool2'):
            self.h_pool2 = tf.nn.max_pool(self.h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Fully connected layer 1 --
        # after 2 round of down sampling, our 28x28 image is down to 7x7x64 feature maps -- maps this to 1024 features.
        with tf.name_scope('fc1'):
            self.W_fc1 = self.weight_variable([7 * 7 * 64, 1024], "W_fc1")
            self.b_fc1 = self.bias_variable([1024], "b_fc1")
            self.h_pool2_flat = tf.reshape(self.h_pool2, [-1, 7 * 7 * 64])
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)
        # Dropout - controls the complexity of the model, prevents co-adaptation of features.
        with tf.name_scope('dropout'):
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)
        # Map the 1024 features to 10 classes, one for each digit
        with tf.name_scope('fc2'):
            self.W_fc2 = self.weight_variable([1024, 10], "W_fc2")
            self.b_fc2 = self.bias_variable([10], "b_fc2")
            self.y_conv = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2

        with tf.name_scope('loss'):
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv)
            self.cross_entropy = tf.reduce_mean(self.cross_entropy)
        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)
        with tf.name_scope('accuracy_check'):
            self.correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
            self.correct_prediction = tf.cast(self.correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(self.correct_prediction)
        with tf.name_scope('predict_number'):
            self.pre_tensor = tf.argmax(self.y_conv, 1)
        self.saver = tf.train.Saver()
        self.sess = sess

    def train(self, data_set, epoch=200):
        self.sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            batch = data_set.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = self.sess.run(self.accuracy, feed_dict={
                    self.x: batch[0], self.y_: batch[1], self.keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))
            self.sess.run(self.train_step, feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})

        print('test accuracy %g' % self.accuracy_rate(data_set))

    def accuracy_rate(self, data_set):
        return self.sess.run(self.accuracy,
                             {self.x: data_set.test.images, self.y_: data_set.test.labels, self.keep_prob: 1.0})

    def predict(self, img):
        #result = self.sess.run(tf.nn.softmax(self.y_conv), feed_dict={self.x: img, self.keep_prob: 1.0})
        #result = self.sess.run(tf.argmax(tf.nn.softmax(self.y_conv), 1), feed_dict={self.x: img, self.keep_prob: 1.0})
        result = self.sess.run(self.y_conv, feed_dict={self.x: img, self.keep_prob: 1.0})
        return result

    def save(self, save_path, save_file_name="model"):
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.mkdir(save_path)
        save_full_path = os.path.join(save_path, save_file_name)
        return self.saver.save(self.sess, save_full_path)

    def restore(self, save_path, save_file_name="model"):
        if os.path.exists(save_path):
            save_full_path = os.path.join(save_path, save_file_name)
            if os.path.isfile(str(save_full_path + ".index")):
                self.saver.restore(self.sess, save_full_path)
            else:
                print("save_file_name is wrong")
        else:
            print("save_path is not found")

    @staticmethod
    def weight_variable(shape, name=None):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name)

    @staticmethod
    def bias_variable(shape, name=None):
        """bias_variable generates a bias variable of a given shape."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
