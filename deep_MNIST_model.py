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

import tensorflow as tf

FLAGS = None


class DeepNeuralNetwork:

    def __init__(self):
        self.W_conv1 = self.weight_variable([5, 5, 1, 32], "W_conv1")
        self.b_conv1 = self.bias_variable([32], "b_conv1")
        self.W_conv2 = self.weight_variable([5, 5, 32, 64], "W_conv2")
        self.b_conv2 = self.bias_variable([64], "b_conv2")
        self.W_fc1 = self.weight_variable([7 * 7 * 64, 1024], "W_fc1")
        self.b_fc1 = self.bias_variable([1024], "b_fc1")
        self.W_fc2 = self.weight_variable([1024, 10], "W_fc2")
        self.b_fc2 = self.bias_variable([10], "b_fc2")

    def deepnn(self, x):
        """deepnn builds the graph for a deep net for classifying digits.
        Args:
          x: an input tensor with the dimensions (N_examples, 784), where 784 is the
          number of pixels in a standard MNIST image.
        Returns:
          A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
          equal to the logits of classifying the digit into one of 10 classes (the
          digits 0-9). keep_prob is a scalar placeholder for the probability of
          dropout.
        """
        # Reshape to use within a convolutional neural net.
        # Last dimension is for "features" - there is only one here, since images are
        # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.

        with tf.name_scope('reshape'):
            x_image = tf.reshape(x, [-1, 28, 28, 1])

        # First convolutional layer - maps one grayscale image to 32 feature maps.
        with tf.name_scope('conv1'):
            h_conv1 = tf.nn.relu(self.conv2d(x_image, self.W_conv1) + self.b_conv1)
            tf.add_to_collection("vars", self.W_conv1)
            tf.add_to_collection("vars", self.b_conv1)

        # Pooling layer - downsamples by 2X.
        with tf.name_scope('pool1'):
            h_pool1 = self.max_pool_2x2(h_conv1)

        # Second convolutional layer -- maps 32 feature maps to 64.
        with tf.name_scope('conv2'):
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, self.W_conv2) + self.b_conv2)
            tf.add_to_collection("vars", self.W_conv2)
            tf.add_to_collection("vars", self.b_conv2)

        # Second pooling layer.
        with tf.name_scope('pool2'):
            h_pool2 = self.max_pool_2x2(h_conv2)

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features.
        with tf.name_scope('fc1'):
            tf.add_to_collection("vars", self.W_fc1)
            tf.add_to_collection("vars", self.b_fc1)

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)

        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Map the 1024 features to 10 classes, one for each digit
        with tf.name_scope('fc2'):
            tf.add_to_collection("vars", self.W_fc2)
            tf.add_to_collection("vars", self.b_fc2)

            y_conv = tf.matmul(h_fc1_drop, self.W_fc2) + self.b_fc2
        return y_conv, keep_prob

    @staticmethod
    def conv2d(x, W):
        """conv2d returns a 2d convolution layer with full stride."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    @staticmethod
    def max_pool_2x2(x):
        """max_pool_2x2 downsamples a feature map by 2X."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

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
