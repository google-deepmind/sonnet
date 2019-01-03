# Copyright 2017 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Gets MNIST or CIFAR10 dataset.


MNIST: Handwritten digits dataset in grayscale images.
CIFAR10: Dataset of 50,000 32x32 color training images, labeled over 10
          categories, and 10,000 test images.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import tensorflow as tf


def get_data(name, train_batch_size, test_batch_size):
  """Gets training and testing dataset iterators.

  Args:
    name: String. Name of dataset, either 'mnist' or 'cifar10'.
    train_batch_size: Integer. Batch size for training.
    test_batch_size: Integer. Batch size for testing.

  Returns:
    Dict containing:
      train_iterator: A tf.data.Iterator, over training data.
      test_iterator: A tf.data.Iterator, over test data.
      num_classes: Integer. Number of class labels.
  """
  if name not in ['mnist', 'cifar10']:
    raise ValueError(
        'Expected dataset \'mnist\' or \'cifar10\', but got %s' % name)
  dataset = getattr(tf.keras.datasets, name)
  num_classes = 10

  # Extract the raw data.
  raw_data = dataset.load_data()
  (images_train, labels_train), (images_test, labels_test) = raw_data

  # Normalize inputs and fix types.
  images_train = images_train.astype(np.float32) / 255.
  images_test = images_test.astype(np.float32) / 255.
  labels_train = labels_train.astype(np.int32).squeeze()
  labels_test = labels_test.astype(np.int32).squeeze()

  # Add a dummy 'color channel' dimension if it is not present.
  if images_train.ndim == 3:
    images_train = np.expand_dims(images_train, -1)
    images_test = np.expand_dims(images_test, -1)

  # Put the data onto the graph as constants.
  train_data = tf.data.Dataset.from_tensor_slices((images_train, labels_train))
  test_data = tf.data.Dataset.from_tensor_slices((images_test, labels_test))

  # Create iterators for each dataset.
  train_iterator = (
      train_data
      # Note: For larger datasets e.g. ImageNet, it will not be feasible to have
      # a shuffle buffer this large.
      .shuffle(buffer_size=len(images_train))
      .batch(train_batch_size)
      .repeat()
      .make_one_shot_iterator()
  )
  test_iterator = test_data.batch(test_batch_size).make_initializable_iterator()
  return dict(
      train_iterator=train_iterator,
      test_iterator=test_iterator,
      num_classes=num_classes)
