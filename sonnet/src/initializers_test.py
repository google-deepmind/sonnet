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
"""Tests for sonnet.v2.src.initializers."""

import itertools

from absl.testing import parameterized
import numpy as np
from sonnet.src import initializers
from sonnet.src import test_utils
import tensorflow as tf


class InitializersTest(test_utils.TestCase, parameterized.TestCase):

  def assertDifferentInitializerValues(self,
                                       init,
                                       shape=None,
                                       dtype=tf.float32):
    if shape is None:
      shape = (100,)
    t1 = self.evaluate(init(shape, dtype))
    t2 = self.evaluate(init(shape, dtype))
    self.assertEqual(t1.shape, shape)
    self.assertEqual(t2.shape, shape)
    self.assertFalse(np.allclose(t1, t2, rtol=1e-15, atol=1e-15))

  def assertRange(self,
                  init,
                  shape,
                  target_mean=None,
                  target_std=None,
                  target_max=None,
                  target_min=None,
                  dtype=tf.float32):
    output = self.evaluate(init(shape, dtype))
    self.assertEqual(output.shape, shape)
    lim = 4e-2
    if target_std is not None:
      self.assertNear(output.std(), target_std, err=lim)
    if target_mean is not None:
      self.assertNear(output.mean(), target_mean, err=lim)
    if target_max is not None:
      self.assertNear(output.max(), target_max, err=lim)
    if target_min is not None:
      self.assertNear(output.min(), target_min, err=lim)


class ConstantInitializersTest(InitializersTest):

  @parameterized.parameters(tf.float32, tf.int32)
  def testZeros(self, dtype):
    self.assertRange(
        initializers.Zeros(),
        shape=(4, 5),
        target_mean=0.,
        target_max=0.,
        dtype=dtype)

  @parameterized.parameters(tf.float32, tf.int32)
  def testOnes(self, dtype):
    self.assertRange(
        initializers.Ones(),
        shape=(4, 5),
        target_mean=1.,
        target_max=1.,
        dtype=dtype)

  @parameterized.named_parameters(
      ("Tensor", lambda: tf.constant([1.0, 2.0, 3.0]), "Tensor"),
      ("Variable", lambda: tf.Variable([3.0, 2.0, 1.0]), "Variable"),
      ("List", lambda: [], "list"), ("Tuple", lambda: (), "tuple"))
  def testConstantInvalidValue(self, value, value_type):
    with self.assertRaisesRegex(
        TypeError, r"Invalid type for value: .*{}.*".format(value_type)):
      initializers.Constant(value())

  @parameterized.parameters((42, tf.float32), (42.0, tf.float32),
                            (42, tf.int32))
  def testConstantValidValue(self, value, dtype):
    self.assertRange(
        initializers.Constant(value),
        shape=(4, 5),
        target_mean=42.,
        target_max=42.,
        dtype=dtype)

  @parameterized.parameters(initializers.Zeros, initializers.Ones)
  def testInvalidDataType(self, initializer):
    init = initializer()
    with self.assertRaisesRegex(
        ValueError, r"Expected integer or floating point type, got "):
      init([1], dtype=tf.string)

  def testInvalidDataTypeConstant(self):
    init = initializers.Constant(0)
    with self.assertRaisesRegex(
        ValueError, r"Expected integer or floating point type, got "):
      init([1], dtype=tf.string)

  def testTFFunction(self):
    init = initializers.Constant(2)
    f = tf.function(lambda t: init(tf.shape(t), t.dtype))

    expected = init([7, 4], tf.float32)
    x = f(tf.zeros([7, 4]))
    self.assertAllEqual(expected, x)

  def testBatchAgnostic(self):
    init = initializers.Constant(2)
    spec = tf.TensorSpec(shape=[None, None])
    f = tf.function(lambda t: init(tf.shape(t), t.dtype))
    f = f.get_concrete_function(spec)

    expected = init([7, 4], tf.float32)
    x = f(tf.ones([7, 4]))
    self.assertAllEqual(expected, x)


class RandomUniformInitializerTest(InitializersTest):

  def testRangeInitializer(self):
    shape = (16, 8, 128)
    self.assertRange(
        initializers.RandomUniform(minval=-1., maxval=1., seed=124.),
        shape,
        target_mean=0.,
        target_max=1,
        target_min=-1)

  @parameterized.parameters(tf.float32, tf.int32)
  def testDifferentInitializer(self, dtype):
    init = initializers.RandomUniform(0, 10)
    self.assertDifferentInitializerValues(init, dtype=dtype)

  def testInvalidDataType(self):
    init = initializers.RandomUniform()
    with self.assertRaisesRegex(
        ValueError, r"Expected integer or floating point type, got "):
      init([1], dtype=tf.string)

  def testTFFunction(self):
    init = initializers.RandomUniform(seed=42)
    f = tf.function(lambda t: init(tf.shape(t), t.dtype))

    expected = init([7, 4], tf.float32)
    x = f(tf.zeros([7, 4]))
    self.assertEqual(x.shape, [7, 4])
    if self.primary_device != "TPU":  # Seeds don't work as expected on TPU
      self.assertAllEqual(expected, x)

  def testBatchAgnostic(self):
    init = initializers.RandomUniform(seed=42)
    spec = tf.TensorSpec(shape=[None, None])
    f = tf.function(lambda t: init(tf.shape(t), t.dtype))
    f = f.get_concrete_function(spec)

    expected = init([7, 4], tf.float32)
    x = f(tf.ones([7, 4]))
    self.assertEqual(x.shape, [7, 4])
    if self.primary_device != "TPU":  # Seeds don't work as expected on TPU
      self.assertAllEqual(expected, x)


class RandomNormalInitializerTest(InitializersTest):

  def testRangeInitializer(self):
    self.assertRange(
        initializers.RandomNormal(mean=0, stddev=1, seed=153),
        shape=(16, 8, 128),
        target_mean=0.,
        target_std=1)

  def testDifferentInitializer(self):
    init = initializers.RandomNormal(0.0, 1.0)
    self.assertDifferentInitializerValues(init)

  @parameterized.parameters(tf.int32, tf.string)
  def testInvalidDataType(self, dtype):
    init = initializers.RandomNormal(0.0, 1.0)
    with self.assertRaisesRegex(ValueError,
                                r"Expected floating point type, got "):
      init([1], dtype=dtype)

  def testTFFunction(self):
    init = initializers.RandomNormal(seed=42)
    f = tf.function(lambda t: init(tf.shape(t), t.dtype))

    expected = init([7, 4], tf.float32)
    x = f(tf.zeros([7, 4]))
    self.assertEqual(x.shape, [7, 4])
    if self.primary_device != "TPU":  # Seeds don't work as expected on TPU
      self.assertAllEqual(expected, x)

  def testBatchAgnostic(self):
    init = initializers.RandomNormal(seed=42)
    spec = tf.TensorSpec(shape=[None, None])
    f = tf.function(lambda t: init(tf.shape(t), t.dtype))
    f = f.get_concrete_function(spec)

    expected = init([7, 4], tf.float32)
    x = f(tf.ones([7, 4]))
    self.assertEqual(x.shape, [7, 4])
    if self.primary_device != "TPU":  # Seeds don't work as expected on TPU
      self.assertAllEqual(expected, x)


class TruncatedNormalInitializerTest(InitializersTest):

  def testRangeInitializer(self):
    self.assertRange(
        initializers.TruncatedNormal(mean=0, stddev=1, seed=126),
        shape=(16, 8, 128),
        target_mean=0.,
        target_max=2,
        target_min=-2)

  def testDifferentInitializer(self):
    init = initializers.TruncatedNormal(0.0, 1.0)
    self.assertDifferentInitializerValues(init)

  @parameterized.parameters(tf.int32, tf.string)
  def testInvalidDataType(self, dtype):
    init = initializers.TruncatedNormal(0.0, 1.0)
    with self.assertRaisesRegex(ValueError,
                                r"Expected floating point type, got "):
      init([1], dtype=dtype)

  def testTFFunction(self):
    init = initializers.TruncatedNormal(seed=42)
    f = tf.function(lambda t: init(tf.shape(t), t.dtype))

    expected = init([7, 4], tf.float32)
    x = f(tf.zeros([7, 4]))
    self.assertEqual(x.shape, [7, 4])
    if self.primary_device != "TPU":  # Seeds don't work as expected on TPU
      self.assertAllEqual(expected, x)

  def testBatchAgnostic(self):
    init = initializers.TruncatedNormal(seed=42)
    spec = tf.TensorSpec(shape=[None, None])
    f = tf.function(lambda t: init(tf.shape(t), t.dtype))
    f = f.get_concrete_function(spec)

    expected = init([7, 4], tf.float32)
    x = f(tf.ones([7, 4]))
    self.assertEqual(x.shape, [7, 4])
    if self.primary_device != "TPU":  # Seeds don't work as expected on TPU
      self.assertAllEqual(expected, x)


class IdentityInitializerTest(InitializersTest):

  @parameterized.parameters(
      *itertools.product([(4, 5), (3, 3), (3, 4, 5),
                          (6, 2, 3, 3)], [3, 1], [tf.float32, tf.int32]))
  def testRange(self, shape, gain, dtype):
    if self.primary_device == "GPU" and dtype == tf.int32:
      self.skipTest("tf.int32 not supported on GPU")

    self.assertRange(
        initializers.Identity(gain),
        shape=shape,
        target_mean=gain / shape[-1],
        target_max=gain,
        dtype=dtype)

  def testInvalidDataType(self):
    init = initializers.Identity()
    with self.assertRaisesRegex(
        ValueError, r"Expected integer or floating point type, got "):
      init([1, 2], dtype=tf.string)

  @parameterized.parameters(tf.float32, tf.int32)
  def testInvalidShape(self, dtype):
    init = initializers.Identity()
    with self.assertRaisesRegex(
        ValueError,
        "The tensor to initialize must be at least two-dimensional"):
      init([1], dtype=dtype)

  def testTFFunction(self):
    init = initializers.Identity()
    f = tf.function(lambda t: init(tf.shape(t), t.dtype))

    expected = init([4, 4], tf.float32)
    x = f(tf.ones([4, 4]))
    self.assertAllEqual(expected, x)

  def testTFFunction4D(self):
    init = initializers.Identity()
    f = tf.function(lambda t: init(tf.shape(t), t.dtype))

    expected = init([4, 4, 3, 2], tf.float32)
    x = f(tf.ones([4, 4, 3, 2]))
    self.assertAllEqual(expected, x)

  def testBatchAgnostic(self):
    init = initializers.Identity()
    spec = tf.TensorSpec(shape=[None, None])
    f = tf.function(lambda t: init(tf.shape(t), t.dtype))
    f = f.get_concrete_function(spec)

    expected = init([7, 4], tf.float32)
    x = f(tf.ones([7, 4]))
    self.assertAllEqual(expected, x)


class OrthogonalInitializerTest(InitializersTest):

  def testRangeInitializer(self):
    self.assertRange(
        initializers.Orthogonal(seed=123), shape=(20, 20), target_mean=0.)

  def testDuplicatedInitializer(self):
    init = initializers.Orthogonal()
    self.assertDifferentInitializerValues(init, (10, 10))

  @parameterized.parameters(tf.int32, tf.string)
  def testInvalidDataType(self, dtype):
    init = initializers.Orthogonal()
    with self.assertRaisesRegex(ValueError,
                                r"Expected floating point type, got "):
      init([1, 2], dtype=dtype)

  def testInvalidShape(self):
    init = initializers.Orthogonal()
    with self.assertRaisesRegex(
        ValueError,
        "The tensor to initialize must be at least two-dimensional"):
      init([1], tf.float32)

  @parameterized.named_parameters(
      ("Square", (10, 10)), ("3DSquare", (100, 5, 5)),
      ("3DRectangle", (10, 9, 8)), ("TallRectangle", (50, 40)),
      ("WideRectangle", (40, 50)))
  def testShapesValues(self, shape):
    init = initializers.Orthogonal()
    tol = 1e-5

    t = self.evaluate(init(shape, tf.float32))
    self.assertAllEqual(tuple(shape), t.shape)
    # Check orthogonality by computing the inner product
    t = t.reshape((np.prod(t.shape[:-1]), t.shape[-1]))
    if t.shape[0] > t.shape[1]:
      self.assertAllClose(
          np.dot(t.T, t), np.eye(t.shape[1]), rtol=tol, atol=tol)
    else:
      self.assertAllClose(
          np.dot(t, t.T), np.eye(t.shape[0]), rtol=tol, atol=tol)

  def testTFFunctionSimple(self):
    init = initializers.Orthogonal(seed=42)
    f = tf.function(init)

    x = f([4, 4], tf.float32)
    self.assertAllEqual(x.shape, [4, 4])

  def testTFFunction(self):
    if self.primary_device == "TPU":
      self.skipTest("Dynamic slice not supported on TPU")

    init = initializers.Orthogonal(seed=42)
    f = tf.function(lambda t: init(tf.shape(t), t.dtype))

    expected = init([4, 4], tf.float32)
    x = f(tf.ones([4, 4]))
    self.assertAllEqual(expected, x)

  def testBatchAgnostic(self):
    if self.primary_device == "TPU":
      self.skipTest("Dynamic slice not supported on TPU")

    init = initializers.Orthogonal(seed=42)
    spec = tf.TensorSpec(shape=[None, None])
    f = tf.function(lambda t: init(tf.shape(t), t.dtype))
    f = f.get_concrete_function(spec)

    expected = init([7, 4], tf.float32)
    x = f(tf.ones([7, 4]))
    self.assertAllEqual(expected, x)


class VarianceScalingInitializerTest(InitializersTest):

  def testTruncatedNormalDistribution(self):
    shape = (100, 100)
    init = initializers.VarianceScaling(distribution="truncated_normal")

    self.assertRange(
        init, shape=shape, target_mean=0., target_std=1. / np.sqrt(shape[0]))

  def testNormalDistribution(self):
    shape = (100, 100)
    init = initializers.VarianceScaling(distribution="normal")

    self.assertRange(
        init, shape=shape, target_mean=0., target_std=1. / np.sqrt(shape[0]))

  def testUniformDistribution(self):
    shape = (100, 100)
    init = initializers.VarianceScaling(distribution="uniform")

    self.assertRange(
        init, shape=shape, target_mean=0., target_std=1. / np.sqrt(shape[0]))

  def testGlorotUniform(self):
    shape = (5, 6, 4, 2)
    fan_in, fan_out = initializers._compute_fans(shape)
    std = np.sqrt(2. / (fan_in + fan_out))
    self.assertRange(
        initializers.VarianceScaling(
            scale=1.0, mode="fan_avg", distribution="uniform", seed=123),
        shape,
        target_mean=0.,
        target_std=std)

  def test_GlorotNormal(self):
    shape = (5, 6, 4, 2)
    fan_in, fan_out = initializers._compute_fans(shape)
    std = np.sqrt(2. / (fan_in + fan_out))
    self.assertRange(
        initializers.VarianceScaling(
            scale=1.0,
            mode="fan_avg",
            distribution="truncated_normal",
            seed=123),
        shape,
        target_mean=0.,
        target_std=std)

  def testLecunUniform(self):
    shape = (5, 6, 4, 2)
    fan_in, _ = initializers._compute_fans(shape)
    std = np.sqrt(1. / fan_in)
    self.assertRange(
        initializers.VarianceScaling(
            scale=1.0, mode="fan_in", distribution="uniform", seed=123),
        shape,
        target_mean=0.,
        target_std=std)

  def testLecunNormal(self):
    shape = (5, 6, 4, 2)
    fan_in, _ = initializers._compute_fans(shape)
    std = np.sqrt(1. / fan_in)
    self.assertRange(
        initializers.VarianceScaling(
            scale=1.0, mode="fan_in", distribution="truncated_normal",
            seed=123),
        shape,
        target_mean=0.,
        target_std=std)

  def testHeUniform(self):
    shape = (5, 6, 4, 2)
    fan_in, _ = initializers._compute_fans(shape)
    std = np.sqrt(2. / fan_in)
    self.assertRange(
        initializers.VarianceScaling(
            scale=2.0, mode="fan_in", distribution="uniform", seed=123),
        shape,
        target_mean=0.,
        target_std=std)

  def testHeNormal(self):
    shape = (5, 6, 4, 2)
    fan_in, _ = initializers._compute_fans(shape)
    std = np.sqrt(2. / fan_in)
    self.assertRange(
        initializers.VarianceScaling(
            scale=2.0, mode="fan_in", distribution="truncated_normal",
            seed=123),
        shape,
        target_mean=0.,
        target_std=std)

  @parameterized.parameters(
      itertools.product(["fan_in", "fan_out", "fan_avg"],
                        ["uniform", "truncated_normal", "normal"]))
  def testMixedShape(self, mode, distribution):
    init = initializers.VarianceScaling(mode=mode, distribution=distribution)
    tf.random.set_seed(42)
    x = init([tf.constant(4), 2], tf.float32)
    tf.random.set_seed(42)
    expected = init([4, 2], tf.float32)
    self.assertEqual(x.shape, [4, 2])
    if self.primary_device != "TPU":  # Seeds don't work as expected on TPU
      self.assertAllEqual(expected, x)

  @parameterized.parameters(
      itertools.product(["fan_in", "fan_out", "fan_avg"],
                        ["uniform", "truncated_normal", "normal"]))
  def testWithTFFunction(self, mode, distribution):
    init = initializers.VarianceScaling(
        mode=mode, distribution=distribution, seed=42)
    f = tf.function(lambda t: init(tf.shape(t), t.dtype))
    x = f(tf.zeros([4, 2]))
    expected = init([4, 2], tf.float32)
    self.assertEqual(x.shape, [4, 2])
    if self.primary_device != "TPU":  # Seeds don't work as expected on TPU
      self.assertAllClose(expected, x)

  @parameterized.parameters(
      itertools.product(["fan_in", "fan_out", "fan_avg"],
                        ["uniform", "truncated_normal", "normal"]))
  def testBatchAgnostic(self, mode, distribution):
    init = initializers.VarianceScaling(
        mode=mode, distribution=distribution, seed=42)
    spec = tf.TensorSpec(shape=[None, None])
    f = tf.function(lambda t: init(tf.shape(t), t.dtype))
    f = f.get_concrete_function(spec)

    expected = init([7, 4], tf.float32)
    x = f(tf.ones([7, 4]))
    self.assertEqual(x.shape, [7, 4])
    if self.primary_device != "TPU":  # Seeds don't work as expected on TPU
      self.assertAllClose(expected, x)

  @parameterized.parameters(tf.int32, tf.string)
  def testInvalidDataType(self, dtype):
    init = initializers.VarianceScaling()
    with self.assertRaisesRegex(ValueError,
                                r"Expected floating point type, got "):
      init([1, 2], dtype=dtype)

  def testCheckInitializersInvalidType(self):
    with self.assertRaisesRegex(TypeError,
                                "Initializers must be a dict-like object."):
      initializers.check_initializers([1, 2, 3], ("a"))

  def testCheckInitalizersEmpty(self):
    a = initializers.check_initializers(None, ("b"))
    self.assertEqual(a, {})

  @parameterized.named_parameters(("Tuple", ("a", "b")), ("List", ["a", "b"]),
                                  ("Set", {"a", "b"}))
  def testCheckInitalizersValid(self, keys):
    initializers.check_initializers({
        "a": lambda x, y: 0,
        "b": lambda x, y: 1
    }, keys)

  def testCheckInitalizersInvalid(self):
    with self.assertRaisesRegex(
        KeyError,
        r"Invalid initializer keys 'a', initializers can only be provided for"):
      initializers.check_initializers({
          "a": lambda x, y: 0,
          "b": lambda x, y: 1
      }, ("b"))


if __name__ == "__main__":
  tf.test.main()
