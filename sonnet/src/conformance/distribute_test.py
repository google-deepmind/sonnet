# Copyright 2019 The Sonnet Authors. All Rights Reserved.
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
"""Tests Sonnet and TF Distribution Strategy."""

from typing import Callable, Tuple

from absl.testing import parameterized
import sonnet as snt
from sonnet.src import test_utils
from sonnet.src.conformance import descriptors
from sonnet.src.conformance import goldens
from sonnet.src.distribute import replicator as snt_replicator
from sonnet.src.distribute import replicator_test_utils as replicator_utils
import tensorflow as tf


class TpuReplicatorTest(test_utils.TestCase, parameterized.TestCase):

  @test_utils.combined_named_parameters(goldens.named_goldens(),
                                        replicator_utils.named_replicators())
  def test_variable_creation_in_replica_context(self, golden, replicator_fn):
    tf.random.set_seed(None)
    replicator = replicator_fn()

    with replicator.scope():
      mod = golden.create_module()

    @tf.function
    def forward():
      step = lambda: golden.create_all_variables(mod)
      return replicator.run(step)

    # TODO(b/132329316) Remove when `xla.compile` allows tf.device(TPU).
    with tf.device(None):
      variables_per_replica = forward()

    self.assertLen(variables_per_replica, golden.num_variables)

    for per_replica_variable in variables_per_replica:
      self.assertSameValuePerReplica(replicator, per_replica_variable)

  def assertSameValuePerReplica(self, replicator, per_replica):
    per_replica = replicator.experimental_local_results(per_replica)
    first_replica = per_replica[0]
    for nth_replica in per_replica[1:]:
      self.assertAllEqual(first_replica, nth_replica)

  @test_utils.combined_named_parameters(descriptors.RNN_CORES,
                                        test_utils.named_bools("dynamic"),
                                        replicator_utils.named_replicators())
  def test_unroll(
      self,
      core_fn: Callable[[], snt.RNNCore],
      input_shape: Tuple[int],
      dtype: tf.DType,
      dynamic: bool,
      replicator_fn: tf.distribute.Strategy,
  ):
    replicator = replicator_fn()
    with replicator.scope():
      core = core_fn()

    def step_fn():
      def forward():
        unroll = snt.dynamic_unroll if dynamic else snt.static_unroll
        sequence = tf.ones((1,) + input_shape, dtype)
        state = core.initial_state(input_shape[0])
        return unroll(core, sequence, state)

      return replicator.run(forward)

    # TpuReplicator doesn't support pure eager mode.
    if isinstance(replicator, snt_replicator.TpuReplicator):
      step_fn = tf.function(step_fn)

    # TODO(b/132329316) Remove when `xla.compile` allows tf.device(TPU).
    with tf.device(None):
      out_sequence, final_state = step_fn()

    self.assertSameValuePerReplica(replicator, out_sequence)
    self.assertSameValuePerReplica(replicator, final_state)

if __name__ == "__main__":
  tf.test.main()
