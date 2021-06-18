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
"""Tests for sonnet.v2.src.linear."""

from absl.testing import parameterized
import numpy as np
from sonnet.src import linear
from sonnet.src import test_utils
import tensorflow as tf


class LinearTest(test_utils.TestCase, parameterized.TestCase):

  def testInitW(self):
    my_initializer = lambda shape, dtype: None
    mod = linear.Linear(1, w_init=my_initializer)
    self.assertIs(mod.w_init, my_initializer)

  def testInitB(self):
    my_initializer = lambda shape, dtype: None
    mod = linear.Linear(1, b_init=my_initializer)
    self.assertIs(mod.b_init, my_initializer)

  def testInitializerKeysInvalidWithoutBias(self):
    with self.assertRaisesRegex(ValueError, "b_init must be None"):
      linear.Linear(1, with_bias=False, b_init=tf.zeros_initializer())

  def testParametersCreatedOnce(self):
    mod = linear.Linear(1)
    mod(tf.constant([[1.]]))
    w, b = mod.w, mod.b
    mod(tf.constant([[1.]]))
    self.assertIs(mod.w, w)
    self.assertIs(mod.b, b)

  def testParameterShape(self):
    batch_size = 1
    input_size = 2
    output_size = 3
    mod = linear.Linear(output_size)
    mod(tf.ones([batch_size, input_size]))
    self.assertEqual(mod.w.shape.as_list(), [input_size, output_size])
    self.assertEqual(mod.b.shape.as_list(), [output_size])

  @parameterized.parameters([tf.float16, tf.float32, tf.int32])
  def testParameterDtype(self, dtype):
    if dtype == tf.int32 and self.primary_device in ("GPU", "TPU"):
      self.skipTest("int32 not supported on %s" % self.primary_device)
    elif self.primary_device == "TPU" and dtype == tf.float16:
      dtype = tf.bfloat16

    mod = linear.Linear(1, w_init=tf.zeros_initializer())
    out = mod(tf.ones([1, 1], dtype=dtype))
    self.assertEqual(out.dtype, dtype)
    self.assertEqual(mod.w.dtype, dtype)
    self.assertEqual(mod.b.dtype, dtype)

  def testBiasZeroInitialized(self):
    mod = linear.Linear(1)
    mod(tf.constant([[1.]]))
    self.assertEqual(mod.b.numpy(), [0.])

  def testCall(self):
    batch_size = 1
    input_size = 2
    output_size = 3

    def numpy_linear():
      w = np.ndarray([input_size, output_size], dtype=np.float32)
      w.fill(2.)
      b = np.ndarray([output_size], dtype=np.float32)
      b.fill(3.)
      i = np.ones([batch_size, input_size], dtype=np.float32)
      return np.matmul(i, w) + b

    l = linear.Linear(
        output_size,
        w_init=tf.constant_initializer(2.),
        b_init=tf.constant_initializer(3.))
    tf_output = l(tf.ones([batch_size, input_size]))
    self.assertAllEqual(tf_output, numpy_linear())

  def testCallMultiBatch(self):
    l = linear.Linear(5)
    input_tensor = tf.random.uniform([1, 2, 3, 4])
    tf_output = l(input_tensor)

    w_np = l.w.numpy()
    b_np = l.b.numpy()
    input_tensor_np = input_tensor.numpy()
    np_output = np.matmul(input_tensor_np, w_np) + b_np

    # TPU uses bfloat16 internally, so larger deviations are expected.
    self.assertAllClose(tf_output, np_output, atol=1e-2, rtol=5e-2)

  @parameterized.parameters(True, False)
  def testFunction(self, with_bias):
    linear_1 = linear.Linear(
        3, with_bias=with_bias, w_init=tf.ones_initializer())
    linear_2 = linear.Linear(
        3, with_bias=with_bias, w_init=tf.ones_initializer())
    defun_linear = tf.function(linear_2)

    iterations = 5

    for _ in range(iterations):
      x = tf.random.uniform([1, 5])
      y1 = linear_1(x)
      y2 = defun_linear(x)

      self.assertAllClose(self.evaluate(y1), self.evaluate(y2), atol=1e-4)

  def testUnknownBatchSize(self):
    x = tf.TensorSpec([None, 4], dtype=tf.float32)

    l = linear.Linear(3)
    defun_linear = tf.function(l)

    defun_linear.get_concrete_function(x)

    out = defun_linear(tf.ones([2, 4]))
    expected_out = l(tf.ones([2, 4]))
    self.assertEqual(out.shape, [2, 3])
    self.assertAllEqual(self.evaluate(expected_out), self.evaluate(out))

    out = defun_linear(tf.ones([4, 4]))
    self.assertEqual(out.shape, [4, 3])

  def testUnknownInputSize(self):
    x = tf.TensorSpec([None, None], dtype=tf.float32)

    l = linear.Linear(3)
    defun_linear = tf.function(l)

    with self.assertRaisesRegex(
        ValueError, "Input size must be specified at module build time."):
      defun_linear.get_concrete_function(x)

  def testMultiBatchOutputDimensions(self):
    x = tf.TensorSpec([None, None, None, 2], dtype=tf.float32)

    l = linear.Linear(7)
    defun_linear = tf.function(l)

    defun_linear.get_concrete_function(x)

    out = defun_linear(tf.ones([1, 5, 3, 2]))
    expected_out = l(tf.ones([1, 5, 3, 2]))
    self.assertEqual(out.shape, [1, 5, 3, 7])
    self.assertAllEqual(self.evaluate(expected_out), self.evaluate(out))

    out = defun_linear(tf.ones([2, 4, 5, 2]))
    self.assertEqual(out.shape, [2, 4, 5, 7])

  @parameterized.named_parameters(("1D", [1]),)
  def testIncorrectDims(self, shape):
    l = linear.Linear(3)
    with self.assertRaisesRegex(ValueError, "Shape .* must have rank >= 2"):
      l(tf.ones(shape))

  def testInputSize(self):
    batch_size = 1
    input_size = 2
    output_size = 3
    mod = linear.Linear(output_size)
    mod(tf.ones([batch_size, input_size]))
    self.assertEqual(mod.input_size, input_size)

  def testOutputSize(self):
    mod = linear.Linear(1)
    self.assertEqual(mod.output_size, 1)


if __name__ == "__main__":
  tf.test.main()
