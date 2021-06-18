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
"""Tests for sonnet.v2.src.build."""

from sonnet.src import build
from sonnet.src import test_utils
import tensorflow as tf


class BuildTest(test_utils.TestCase):

  def test_call_with_shape_lke_object(self):
    output_spec = build.build(tensor_identity, [1, None, 3])
    self.assertEqual(output_spec, tf.TensorSpec([1, None, 3]))

  def test_output_spec(self):
    dtype = tf.float32 if self.primary_device == "TPU" else tf.float16
    inputs = {"foo": [tf.ones([], dtype), None]}
    output_spec = build.build(lambda x: x, inputs)
    self.assertEqual(output_spec,
                     {"foo": [tf.TensorSpec([], dtype), None]})

  def test_does_not_trigger_sideeffects(self):
    mod = IncrementsCounter()
    output_spec = build.build(mod)
    self.assertIsNone(output_spec)
    self.assertEqual(mod.counter.numpy(), 0)


def tensor_identity(x):
  assert isinstance(x, tf.Tensor)
  return x


class IncrementsCounter(tf.Module):

  def __call__(self):
    if not hasattr(self, "counter"):
      self.counter = tf.Variable(0)
    self.counter.assign_add(1)

if __name__ == "__main__":
  tf.test.main()
