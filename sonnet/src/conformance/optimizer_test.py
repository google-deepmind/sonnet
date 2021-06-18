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
"""Conformance tests for models and optimization."""

from absl.testing import parameterized
from sonnet.src import test_utils
from sonnet.src.conformance import descriptors
import tensorflow as tf

BATCH_MODULES = descriptors.BATCH_MODULES
RECURRENT_MODULES = descriptors.RECURRENT_MODULES


class OptimizerConformanceTest(test_utils.TestCase, parameterized.TestCase):

  @test_utils.combined_named_parameters(
      BATCH_MODULES + RECURRENT_MODULES,
      test_utils.named_bools("construct_module_in_function"),
  )
  def test_variable_order_is_constant(self, module_fn, input_shape, dtype,
                                      construct_module_in_function):
    """Test that variable access order is consistent in built in modules."""
    logged_variables = []
    mod = [None]
    if not construct_module_in_function:
      mod[0] = module_fn()

    x = tf.zeros(input_shape, dtype=dtype)

    @tf.function(autograph=False)
    def f():
      with tf.GradientTape() as tape:
        if not mod[0]:
          mod[0] = module_fn()
        mod[0](x)  # pylint: disable=not-callable

      # Leak out the variables that were used.
      logged_variables.append(
          [(id(v), v.name) for v in tape.watched_variables()])

    # NOTE: This will run `f` twice iff `f` creates params.
    f()

    if len(logged_variables) == 1:
      self.skipTest("Module did not create variables in forward pass.")
    else:
      assert len(logged_variables) == 2
      self.assertCountEqual(logged_variables[0], logged_variables[1])

if __name__ == "__main__":
  tf.test.main()
