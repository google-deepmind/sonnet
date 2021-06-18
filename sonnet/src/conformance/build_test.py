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
"""Tests modules support `snt.build`."""

from absl.testing import parameterized
import sonnet as snt
from sonnet.src import test_utils
from sonnet.src.conformance import descriptors
import tensorflow as tf
import tree

BATCH_MODULES = descriptors.BATCH_MODULES
RECURRENT_MODULES = descriptors.RECURRENT_MODULES


def if_present(f):
  return lambda o: f(o) if o is not None else None


class BuildTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(*(BATCH_MODULES + RECURRENT_MODULES))
  def test_build(self, module_fn, input_shape, dtype):
    module = module_fn()
    build_output_spec = snt.build(module, tf.TensorSpec(input_shape, dtype))
    actual_output = module(tf.ones(input_shape, dtype))
    actual_output_spec = tree.map_structure(
        if_present(lambda t: tf.TensorSpec(t.shape, t.dtype)), actual_output)
    tree.map_structure(self.assertCompatible, build_output_spec,
                       actual_output_spec)

  def assertCompatible(self, a: tf.TensorSpec, b: tf.TensorSpec):
    self.assertTrue(a.shape.is_compatible_with(b.shape))
    self.assertEqual(a.dtype, b.dtype)


if __name__ == "__main__":
  tf.test.main()
