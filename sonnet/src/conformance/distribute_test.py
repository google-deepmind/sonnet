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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
from sonnet.src import replicator_test_utils as replicator_utils
from sonnet.src import test_utils
from sonnet.src.conformance import goldens
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
      return replicator.experimental_run_v2(step)

    # TODO(b/132329316) Remove when `xla.compile` allows tf.device(TPU).
    with tf.device(None):
      variables_per_replica = forward()

    self.assertLen(variables_per_replica, golden.num_variables)

    # Check all variables have the same value on each replica.
    for per_replica in variables_per_replica:
      per_replica = replicator.experimental_local_results(per_replica)
      first_replica = per_replica[0]
      for nth_replica in per_replica[1:]:
        self.assertAllEqual(first_replica, nth_replica)

if __name__ == "__main__":
  # tf.enable_v2_behavior()
  tf.test.main()
