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

from tensorflow.examples.tutorials.mnist import input_data
from deep_MNIST_model_ver2 import DeepNeuralNetwork
import tensorflow as tf
import numpy as np
import cv2

FLAGS = None


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    save_path = "models/deep_MNIST_save/ver2.1"
    image_path = "picture/number_2_resize.png"
    #image_path = "picture/mnist_pre4_cor8.jpg"

    with tf.Session() as sess:
        new_model = DeepNeuralNetwork(sess)
        new_model.restore(save_path)
        print('test accuracy %g' % new_model.accuracy_rate(mnist))

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_arr = np.array(img, dtype=float).reshape(1, -1)
        print(img_arr.dtype)
        print(img_arr.shape)
        predict = new_model.predict(img_arr)
        print(predict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
