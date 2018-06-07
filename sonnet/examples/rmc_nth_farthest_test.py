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
"""Tests for sonnet.examples.rmc_nth_farthest."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
from sonnet.examples import dataset_nth_farthest
from sonnet.examples import rmc_nth_farthest
import tensorflow as tf


class RMCNthFarthestTest(tf.test.TestCase):

  def setUp(self):
    self._batch_size = 2
    self._num_objects = 2
    self._feature_size = 2

  def test_object_sequence_model(self):
    """Test the model class."""
    core = snt.RelationalMemory(
        mem_slots=2, head_size=4, num_heads=1, num_blocks=1, gate_style="unit")
    final_mlp = snt.nets.MLP(
        output_sizes=(5,), activate_final=True)
    model = rmc_nth_farthest.SequenceModel(
        core=core,
        target_size=self._num_objects,
        final_mlp=final_mlp)
    logits = model(tf.zeros(
        (self._batch_size, self._num_objects, self._feature_size)))
    self.assertAllEqual(logits.shape,
                        (self._batch_size, self._num_objects))

  def test_build_and_train(self):
    """Test the example TF graph build."""
    total_iterations = 2
    reporting_interval = 1
    steps, train_losses, test_accs = rmc_nth_farthest.build_and_train(
        total_iterations, reporting_interval, test=True)
    self.assertEqual(len(steps), total_iterations)
    self.assertEqual(len(train_losses), total_iterations)
    self.assertEqual(len(test_accs), total_iterations)

  def test_nth_farthest_datset(self):
    """Test the dataset class."""
    dataset = dataset_nth_farthest.NthFarthest(
        self._batch_size, self._num_objects, self._feature_size)
    inputs, _ = dataset.get_batch()
    final_feature_size = self._feature_size + 3 * self._num_objects
    self.assertAllEqual(
        inputs.shape,
        (self._batch_size, self._num_objects, final_feature_size))

if __name__ == "__main__":
  tf.test.main()
