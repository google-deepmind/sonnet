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
from sonnet.examples import learn_to_execute
from sonnet.examples import rmc_learn_to_execute
import tensorflow as tf


class RMCLearnTest(tf.test.TestCase):

  def setUp(self):
    self._batch_size = 2
    self._seq_sz_in = 10
    self._seq_sz_out = 3
    self._feature_size = 8
    self._nesting = 2
    self._literal_length = 3

  def test_object_sequence_model(self):
    """Test the model class."""
    core = snt.RelationalMemory(
        mem_slots=2, head_size=4, num_heads=1, num_blocks=1, gate_style="unit")
    final_mlp = snt.nets.MLP(
        output_sizes=(5,), activate_final=True)
    model = rmc_learn_to_execute.SequenceModel(
        core=core,
        target_size=self._feature_size,
        final_mlp=final_mlp)
    dummy_in = tf.zeros(
        (self._seq_sz_in, self._batch_size, self._feature_size))
    dummy_out = tf.zeros(
        (self._seq_sz_out, self._batch_size, self._feature_size))
    sizes = tf.ones((self._batch_size))
    logits = model(dummy_in, dummy_out, sizes, sizes)
    self.assertAllEqual(
        logits.shape, (self._seq_sz_out, self._batch_size, self._feature_size))

  def test_build_and_train(self):
    """Test the example TF graph build."""
    total_iterations = 2
    reporting_interval = 1
    rmc_learn_to_execute.build_and_train(
        total_iterations, reporting_interval, test=True)

  def test_learn_to_execute_datset(self):
    """Test the dataset class."""
    dataset = learn_to_execute.LearnToExecute(
        self._batch_size, self._literal_length, self._nesting)
    dataset_iter = dataset.make_one_shot_iterator().get_next()
    logit_size = dataset.state.vocab_size
    seq_sz_in = dataset.state.num_steps
    seq_sz_out = dataset.state.num_steps_out
    self.assertAllEqual(
        dataset_iter[0].shape, (seq_sz_in, self._batch_size, logit_size))
    self.assertAllEqual(
        dataset_iter[1].shape, (seq_sz_out, self._batch_size, logit_size))
    self.assertAllEqual(
        dataset_iter[2].shape, (seq_sz_out, self._batch_size, logit_size))
    self.assertAllEqual(dataset_iter[3].shape, (self._batch_size,))
    self.assertAllEqual(dataset_iter[4].shape, (self._batch_size,))

if __name__ == "__main__":
  tf.test.main()
