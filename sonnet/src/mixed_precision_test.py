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
"""Tests for Mixed Precision."""

from absl.testing import parameterized
from sonnet.src import base
from sonnet.src import mixed_precision
from sonnet.src import test_utils
import tensorflow as tf
import tree


class DummyVar(base.Module, test_utils.TestCase):

  def __init__(self, x):
    super().__init__()
    test_utils.TestCase.__init__(self)
    self.x = x

  def check_type(self, _, dtype):
    # TODO(loreno): handle dictionaries with non-sortable keys and change
    # this test to assertEqual once that works
    self.assertTrue(self.x.dtype == dtype)  # pylint: disable=g-generic-assert
    return self.x

  def check_type_structure(self, _, dtype):
    # pylint: disable=g-generic-assert
    tree.map_structure(lambda y: self.assertTrue(y.dtype == dtype), self.x)
    return self.x

  def runTest(self):
    pass


class DummyInput(test_utils.TestCase):

  def __init__(self, _):
    super().__init__()
    test_utils.TestCase.__init__(self)

  def check_type(self, x, dtype):
    self.assertEqual(x.dtype, dtype)
    return x

  def check_type_structure(self, x, dtype):
    tree.map_structure(lambda y: self.assertEqual(y.dtype, dtype), x)
    return x

  def runTest(self):
    pass


@parameterized.parameters(DummyVar, DummyInput)
class MixedPrecisionClassTest(test_utils.TestCase):

  def test_float16_mode_variable_eligible_class(self, test_class):
    mixed_precision.enable(tf.float32)

    x = tf.Variable([[1., 9.], [5., 0.]])
    d = test_class(x)
    d.check_type = mixed_precision.modes([tf.float32, tf.float16])(d.check_type)

    mixed_precision.enable(tf.float16)
    # First call to forward fn always runs in full precision.
    self.assertEqual(d.check_type(x, tf.float32).dtype, tf.float32)
    # Subsequent calls run in mixed precision.
    self.assertEqual(d.check_type(x, tf.float16).dtype, tf.float32)

  def test_float16_mode_disable_class(self, test_class):
    mixed_precision.enable(tf.float32)

    x = tf.Variable([[1., 9.], [5., 0.]])
    d = test_class(x)
    d.check_type = mixed_precision.modes([tf.float32, tf.float16])(d.check_type)

    mixed_precision.enable(tf.float16)
    self.assertEqual(d.check_type(x, tf.float32).dtype, tf.float32)
    self.assertEqual(d.check_type(x, tf.float16).dtype, tf.float32)
    mixed_precision.disable()
    self.assertEqual(d.check_type(x, tf.float32).dtype, tf.float32)

  def test_float16_mode_nested_eligible_class(self, test_class):
    mixed_precision.enable(tf.float32)

    # TODO(loreno): test nested combo of tensor and Variables once the custom
    # variable getter can cast tensors.
    x = tf.Variable([[1., 9.], [5., 0.]])
    y = tf.Variable([[1., 9.], [8., 9.]])
    z = (x, y)

    d = test_class(z)
    d.check_type_structure = mixed_precision.modes([tf.float32, tf.float16])(
        d.check_type_structure)

    self.assertTrue(tree.is_nested(z))
    mixed_precision.enable(tf.float16)

    first_run = d.check_type_structure(z, tf.float32)
    self.assertEqual(first_run[0].dtype, tf.float32)
    self.assertEqual(first_run[1].dtype, tf.float32)
    second_run = d.check_type_structure(z, tf.float16)
    self.assertEqual(second_run[0].dtype, tf.float32)
    self.assertEqual(second_run[1].dtype, tf.float32)

  def test_float16_mode_eligible_multiple_instances_class(self, test_class):
    mixed_precision.enable(tf.float32)

    x = tf.Variable([[1., 9.], [5., 0.]])
    d = test_class(x)
    d.check_type = mixed_precision.modes([tf.float32, tf.float16])(d.check_type)

    d2 = test_class(x)
    d2.check_type = mixed_precision.modes([tf.float32, tf.float16])(
        d2.check_type)

    mixed_precision.enable(tf.float16)
    self.assertEqual(d.check_type(x, tf.float32).dtype, tf.float32)
    self.assertEqual(d.check_type(x, tf.float16).dtype, tf.float32)
    self.assertEqual(d2.check_type(x, tf.float32).dtype, tf.float32)
    self.assertEqual(d2.check_type(x, tf.float16).dtype, tf.float32)

  def test_float16_mode_ineligible_multiple_instances_class(self, test_class):
    mixed_precision.enable(tf.float32)

    x = tf.Variable([[1., 9.], [5., 0.]])
    d = test_class(x)
    d.check_type = mixed_precision.modes([tf.float32, tf.bfloat16])(
        d.check_type)

    d2 = test_class(x)
    d2.check_type = mixed_precision.modes([tf.float32, tf.bfloat16])(
        d2.check_type)

    mixed_precision.enable(tf.float16)
    self.assertEqual(d.check_type(x, tf.float32).dtype, tf.float32)
    self.assertEqual(d.check_type(x, tf.float32).dtype, tf.float32)
    self.assertEqual(d2.check_type(x, tf.float32).dtype, tf.float32)
    self.assertEqual(d2.check_type(x, tf.float32).dtype, tf.float32)

  def test_float16_mode_multiple_instances_different_eligibility_class(
      self, test_class):
    mixed_precision.enable(tf.float32)

    x = tf.Variable([[1., 9.], [5., 0.]])
    d = test_class(x)
    d.check_type = mixed_precision.modes([tf.float32, tf.bfloat16])(
        d.check_type)

    d2 = test_class(x)
    d2.check_type = mixed_precision.modes([tf.float32, tf.float16])(
        d2.check_type)

    mixed_precision.enable(tf.float16)
    self.assertEqual(d.check_type(x, tf.float32).dtype, tf.float32)
    self.assertEqual(d.check_type(x, tf.float32).dtype, tf.float32)
    self.assertEqual(d2.check_type(x, tf.float32).dtype, tf.float32)
    self.assertEqual(d2.check_type(x, tf.float16).dtype, tf.float32)

  def test_bfloat16_input_float16_mode_eligible_class(self, test_class):
    mixed_precision.enable(tf.float32)

    x = tf.Variable([[1., 9.], [5., 0.]], dtype=tf.bfloat16)
    d = test_class(x)
    d.check_type = mixed_precision.modes([tf.float32, tf.float16])(d.check_type)

    mixed_precision.enable(tf.float16)
    self.assertEqual(d.check_type(x, tf.bfloat16).dtype, tf.bfloat16)
    self.assertEqual(d.check_type(x, tf.float16).dtype, tf.float32)

  def test_float16_input_float32_mode_eligible_class(self, test_class):
    if self.primary_device == 'TPU':
      self.skipTest('float16 not supported on TPU')

    mixed_precision.enable(tf.float32)

    x = tf.Variable([[1., 9.], [5., 0.]], dtype=tf.float16)
    d = test_class(x)
    d.check_type = mixed_precision.modes([tf.float32, tf.float16])(d.check_type)

    self.assertEqual(d.check_type(x, tf.float16).dtype, tf.float16)
    self.assertEqual(d.check_type(x, tf.float32).dtype, tf.float32)

  def test_function_create_module_eligible(self, test_class):
    mixed_precision.enable(tf.float16)

    @mixed_precision.modes([tf.float32, tf.float16])
    def model():
      x = tf.Variable([[1., 9.], [8., 9.]])
      d = test_class(x)
      d.check_type = mixed_precision.modes([tf.float32, tf.float16])(
          d.check_type)

      self.assertEqual(d.check_type(x, tf.float32).dtype, tf.float32)
      self.assertEqual(d.check_type(x, tf.float16).dtype, tf.float32)

    model()

  def test_function_create_module_ineligible(self, test_class):
    mixed_precision.enable(tf.float16)

    @mixed_precision.modes([tf.float32, tf.float16])
    def model():
      x = tf.Variable([[1., 9.], [8., 9.]])
      d = test_class(x)
      d.check_type = mixed_precision.modes([tf.float32, tf.bfloat16])(
          d.check_type)

      self.assertEqual(d.check_type(x, tf.float32).dtype, tf.float32)
      self.assertEqual(d.check_type(x, tf.float32).dtype, tf.float32)

    model()

  def test_function_create_module_not_decorated(self, test_class):
    mixed_precision.enable(tf.float16)

    @mixed_precision.modes([tf.float32, tf.float16])
    def model():
      x = tf.Variable([[1., 9.], [8., 9.]])
      d = test_class(x)
      self.assertEqual(d.check_type(x, tf.float32).dtype, tf.float32)
      self.assertEqual(d.check_type(x, tf.float32).dtype, tf.float32)

    model()

  def test_scoping_option(self, test_class):
    mixed_precision.enable(tf.float32)

    x = tf.Variable([[1., 9.], [8., 9.]])
    d = test_class(x)
    d.check_type = mixed_precision.modes([tf.float32, tf.float16])(d.check_type)

    with mixed_precision.scope(tf.float16):
      self.assertEqual(d.check_type(x, tf.float32).dtype, tf.float32)
      self.assertEqual(d.check_type(x, tf.float16).dtype, tf.float32)

    self.assertEqual(d.check_type(x, tf.float32).dtype, tf.float32)

  def test_scoping_disable(self, test_class):
    mixed_precision.enable(tf.float32)

    x = tf.Variable([[1., 9.], [8., 9.]])
    d = test_class(x)
    d.check_type = mixed_precision.modes([tf.float32, tf.float16])(d.check_type)

    with mixed_precision.scope(tf.float16):
      self.assertEqual(d.check_type(x, tf.float32).dtype, tf.float32)
      self.assertEqual(d.check_type(x, tf.float16).dtype, tf.float32)

      mixed_precision.disable()
      self.assertEqual(d.check_type(x, tf.float32).dtype, tf.float32)

  def test_nested_scoping(self, test_class):
    mixed_precision.enable(tf.float32)

    x = tf.Variable([[1., 9.], [8., 9.]])
    d = test_class(x)
    d.check_type = mixed_precision.modes([tf.float32, tf.float16])(d.check_type)

    with mixed_precision.scope(tf.float16):
      self.assertEqual(d.check_type(x, tf.float32).dtype, tf.float32)
      self.assertEqual(d.check_type(x, tf.float16).dtype, tf.float32)
      with mixed_precision.scope(tf.float32):
        self.assertEqual(d.check_type(x, tf.float32).dtype, tf.float32)
        with mixed_precision.scope(tf.float16):
          self.assertEqual(d.check_type(x, tf.float16).dtype, tf.float32)

    self.assertEqual(d.check_type(x, tf.float32).dtype, tf.float32)


class MixedPrecisionTest(test_utils.TestCase):

  def test_float16_mode_eligible_func(self):
    mixed_precision.enable(tf.float32)
    self.assertEqual(mixed_precision._get_mixed_precision_mode(), tf.float32)

    @mixed_precision.modes([tf.float32, tf.float16])
    def check_type(x, expected_dtype):
      self.assertEqual(x.dtype, expected_dtype)
      return x

    mixed_precision.enable(tf.float16)

    x = tf.Variable([[1., 3], [5., 7.]])
    self.assertEqual(x.dtype, tf.float32)
    self.assertEqual(check_type(x, tf.float32).dtype, tf.float32)
    self.assertEqual(check_type(x, tf.float16).dtype, tf.float32)

  def test_float32_mode_eligible_func(self):
    mixed_precision.enable(tf.float32)
    self.assertEqual(mixed_precision._get_mixed_precision_mode(), tf.float32)

    @mixed_precision.modes([tf.float32, tf.float16])
    def fwd_func(x):
      self.assertEqual(x.dtype, tf.float32)
      return x

    x = tf.Variable([[1., 3], [5., 7.]])
    self.assertEqual(x.dtype, tf.float32)
    self.assertEqual(fwd_func(x).dtype, tf.float32)
    self.assertEqual(fwd_func(x).dtype, tf.float32)

  def test_float16_mode_ineligible_func(self):
    mixed_precision.enable(tf.float32)

    @mixed_precision.modes([tf.float32, tf.bfloat16])
    def fwd_func(x):
      self.assertEqual(x.dtype, tf.float32)
      return x

    x = tf.Variable([[1., 3], [5., 7.]])
    self.assertEqual(x.dtype, tf.float32)

    mixed_precision.enable(tf.float16)
    self.assertEqual(fwd_func(x).dtype, tf.float32)
    self.assertEqual(fwd_func(x).dtype, tf.float32)

  def test_dont_cast_non_floats_func(self):
    mixed_precision.enable(tf.float32)

    @mixed_precision.modes([tf.float32, tf.float16])
    def fwd_func(x):
      self.assertTrue(x.dtype.is_integer)
      return x

    x = tf.Variable([[1, 9], [8, 9]])
    self.assertTrue(x.dtype.is_integer)

    mixed_precision.enable(tf.float16)
    self.assertTrue(fwd_func(x).dtype.is_integer)
    self.assertTrue(fwd_func(x).dtype.is_integer)

  def test_non_tensor_variable_input_no_cast_func(self):
    mixed_precision.enable(tf.float32)

    @mixed_precision.modes([tf.float32, tf.float16])
    def fwd_func(x):
      self.assertEqual(type(x[0][0]), float)
      return x

    x = [[1., 3], [5., 7.]]
    self.assertEqual(type(x[0][0]), float)

    mixed_precision.enable(tf.float16)
    self.assertEqual(type(fwd_func(x)[0][0]), float)
    self.assertEqual(type(fwd_func(x)[0][0]), float)

  def test_float16_mode_enabled_call_function(self):
    mixed_precision.enable(tf.float32)

    class DummyCall(base.Module, test_utils.TestCase):

      def __init__(self):
        super().__init__()
        test_utils.TestCase.__init__(self)
        self.y = tf.Variable([[1., 3], [5., 7.]])

      @mixed_precision.modes([tf.float16, tf.float32])
      def __call__(self, x, dtype):
        # pylint: disable=g-generic-assert
        self.assertTrue(self.y.dtype == dtype)
        self.assertTrue(x.dtype == dtype)
        return x

      def runTest(self):
        pass

    x = tf.Variable([[1., 3], [5., 7.]])
    self.assertEqual(x.dtype, tf.float32)

    d = DummyCall()
    mixed_precision.enable(tf.float16)
    self.assertEqual(d(x, tf.float32).dtype, tf.float32)
    self.assertEqual(d(x, tf.float16).dtype, tf.float32)

  # TODO(loreno): Run this test against custom variable getters once they can
  # handle and cast tensors
  def test_float16_mode_tensor_eligible_class(self):
    mixed_precision.enable(tf.float32)

    x = tf.constant([[1., 9.], [5., 0.]])
    d = DummyInput(x)
    d.check_type = mixed_precision.modes([tf.float32, tf.float16])(d.check_type)

    mixed_precision.enable(tf.float16)
    self.assertEqual(d.check_type(x, tf.float32).dtype, tf.float32)
    self.assertEqual(d.check_type(x, tf.float16).dtype, tf.float32)


if __name__ == '__main__':
  tf.test.main()
