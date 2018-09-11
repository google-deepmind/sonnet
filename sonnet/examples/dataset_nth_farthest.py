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

"""Defines the dataset for generating sequences of n-th farthest problem.

The "N-th Farthest" task is designed to stress a capacity for relational
reasoning across time. Inputs are a sequence of randomly sampled vectors and
targets are answers to a question of the form:

  "What is the n-th farthest vector (in Euclidean distance) from vector `m`?"

where the vector values, their IDs, `n` and `m` are randomly sampled per
sequence.  The model must compute all pairwise distance relations to the
reference vector `m` which it eventually sees at some point in the sequence.
The vector distances must be implicitly sorted to produce an answer and the
model must sort distance relations between vectors.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
from scipy.spatial import distance as spdistance
import six
import tensorflow as tf


class NthFarthest(object):
  """Choose the nth furthest object from the reference."""

  def __init__(self, batch_size, num_objects, num_features):
    self._batch_size = batch_size
    self._num_objects = num_objects
    self._num_features = num_features

  def _get_single_set(self, num_objects, num_features):
    """Generate one input sequence and output label.

    Each sequences of objects has a feature that consists of the feature vector
    for that object plus the encoding for its ID, the reference vector ID and
    the n-th value relative ID for a total feature size of:

      `num_objects` * 3  + `num_features`

    Args:
      num_objects: int. number of objects in the sequence.
      num_features: int. feature size of each object.

    Returns:
      1. np.ndarray (`num_objects`, (`num_features` + 3 * `num_objects`)).
      2. np.ndarray (1,). Output object reference label.
    """
    # Generate random binary vectors
    data = np.random.uniform(-1, 1, size=(num_objects, num_features))

    distances = spdistance.squareform(spdistance.pdist(data))
    distance_idx = np.argsort(distances)

    # Choose random distance
    nth = np.random.randint(0, num_objects)

    # Pick out the nth furthest for each object
    nth_furthest = distance_idx[:, nth]

    # Choose random reference object
    reference = np.random.randint(0, num_objects)

    # Get identity of object that is the nth furthest from reference object
    labels = nth_furthest[reference]

    # Compile data
    object_ids = np.identity(num_objects)
    nth_matrix = np.zeros((num_objects, num_objects))
    nth_matrix[:, nth] = 1
    reference_object = np.zeros((num_objects, num_objects))
    reference_object[:, reference] = 1

    inputs = np.concatenate([data, object_ids, reference_object, nth_matrix],
                            axis=-1)
    inputs = np.random.permutation(inputs)
    labels = np.expand_dims(labels, axis=0)
    return inputs.astype(np.float32), labels.astype(np.float32)

  def _get_batch_data(self, batch_size, num_objects, num_features):
    """Assembles a batch of input tensors and output labels.

    Args:
      batch_size: int. number of sequence batches.
      num_objects: int. number of objects in the sequence.
      num_features: int. feature size of each object.

    Returns:
      1. np.ndarray (`batch_size`, `num_objects`,
                     (`num_features` + 3 * `num_objects`)).
      2. np.ndarray (`batch_size`). Output object reference label.
    """
    all_inputs = []
    all_labels = []
    for _ in six.moves.range(batch_size):
      inputs, labels = self._get_single_set(num_objects, num_features)
      all_inputs += [inputs]
      all_labels += [labels]
    input_data = np.concatenate(all_inputs, axis=0)
    label_data = np.concatenate(all_labels, axis=0)
    return input_data, label_data

  def get_batch(self):
    """Returns set of nth-farthest input tensors and labels.

    Returns:
      1. tf.Tensor (`batch_size`, `num_objects`,
                     (`num_features` + 3 * `num_objects`)).
      2. tf.Tensor (`batch_size`). Output object reference label.
    """
    params = [self._batch_size, self._num_objects, self._num_features]
    inputs, labels = tf.py_func(self._get_batch_data, params,
                                [tf.float32, tf.float32])
    inputs = tf.reshape(inputs, [self._batch_size, self._num_objects,
                                 self._num_features + self._num_objects * 3])
    labels = tf.reshape(labels, [-1])
    return inputs, labels
