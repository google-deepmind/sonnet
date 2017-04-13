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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np

import sonnet as snt
from sonnet.python.modules.nets import dilation
from sonnet.testing import parameterized

import tensorflow as tf


class IdentityKernelInitializerTest(tf.test.TestCase,
                                    parameterized.ParameterizedTestCase):

  @parameterized.NamedParameters(("Rank4", [2, 2]),
                                 ("SquareFilters", [2, 3, 1, 1]),
                                 ("OddHeighAndWidth", [2, 2, 1, 1]),
                                 ("EqualInAndOutChannels", [3, 3, 2, 1]))
  def testInvalidShapes(self, shape):
    with self.assertRaises(ValueError):
      snt.nets.identity_kernel_initializer(shape)

  def testComputation(self):
    with self.test_session() as sess:
      x = sess.run(snt.nets.identity_kernel_initializer([3, 3, 5, 5]))

      # Iterate over elements. Assert that only the middle pixel is on when in
      # and out channels are same.
      it = np.nditer(x, flags=["multi_index"])
      while not it.finished:
        value, idx = it[0], it.multi_index
        (filter_height, filter_width, in_channel, out_channel) = idx
        if (filter_height == 1 and filter_width == 1 and
            in_channel == out_channel):
          self.assertEqual(value, 1)
        else:
          self.assertEqual(value, 0)
        it.iternext()


class NoisyIdentityKernelInitializerTest(tf.test.TestCase,
                                         parameterized.ParameterizedTestCase):

  @parameterized.NamedParameters(("Rank4", [2, 2]),
                                 ("SquareFilters", [2, 3, 1, 1]),
                                 ("OddHeighAndWidth", [2, 2, 1, 1]),
                                 ("InAndOutChannelsAreMultiples", [3, 3, 2, 7]))
  def testInvalidShapes(self, shape):
    with self.assertRaises(ValueError):
      initializer = snt.nets.noisy_identity_kernel_initializer(2)
      initializer(shape)

  def testComputation(self):
    tf.set_random_seed(0)
    with self.test_session() as sess:
      initializer = snt.nets.noisy_identity_kernel_initializer(2, stddev=1e-20)
      x = initializer([3, 3, 4, 8])
      x = tf.reduce_sum(x, axis=[3])
      x_ = sess.run(x)

      # Iterate over elements. After summing over depth, assert that only the
      # middle pixel is on.
      it = np.nditer(x_, flags=["multi_index"])
      while not it.finished:
        value, idx = it[0], it.multi_index
        (filter_height, filter_width, _) = idx
        if filter_height == 1 and filter_width == 1:
          self.assertAllClose(value, 1)
        else:
          self.assertAllClose(value, 0)
        it.iternext()


class DilationTest(tf.test.TestCase, parameterized.ParameterizedTestCase):

  def setUpWithNumOutputClasses(self, num_output_classes, depth=None):
    """Initialize Dilation module and test images.

    Args:
      num_output_classes: int. Number of output classes the dilation module
        should predict per pixel.
      depth: None or int. Input depth of image. If None, same as
        num_output_classes.
    """
    self._num_output_classes = num_output_classes
    self._model_size = "basic"
    self._module = snt.nets.Dilation(
        num_output_classes=self._num_output_classes,
        model_size=self._model_size)

    self._batch_size = 1
    self._height = self._width = 5
    self._depth = depth or num_output_classes

    # Generate images with all-positive values. This means that so long as
    # convolution kernels are initialized to identity operators, applying the
    # network should be an identity operation (negative values get zeroed out by
    # ReLUs).
    self._rng = np.random.RandomState(0)
    self._images = np.abs(
        self._rng.randn(self._batch_size, self._height, self._width,
                        self._depth).astype(np.float32))

  @parameterized.Parameters(1, 3)
  def testShapeInference(self, num_output_classes):
    self.setUpWithNumOutputClasses(num_output_classes)
    x = self._module(tf.convert_to_tensor(self._images))
    self.assertTrue(x.get_shape().is_compatible_with(
        [self._batch_size, self._height, self._width, num_output_classes]))

  @parameterized.Parameters(1, 3)
  def testBasicComputation(self, num_output_classes):
    self.setUpWithNumOutputClasses(num_output_classes)
    x = self._module(tf.convert_to_tensor(self._images))

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      x_ = sess.run(x)

      # Default initialization produces an identity operator.
      self.assertAllClose(x_, self._images)

  @parameterized.Parameters(1, 3)
  def testLargeComputation(self, num_output_classes):
    self.setUpWithNumOutputClasses(
        num_output_classes, depth=3 * num_output_classes)
    self.setUpWithNumOutputClasses(num_output_classes)
    module = snt.nets.Dilation(
        num_output_classes=num_output_classes, model_size="large")
    x = module(tf.convert_to_tensor(self._images))

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      x_ = sess.run(x)

      # Default initialization produces something like an operator, but the
      # number of channels differs. However, summing across channels should
      # recover a near-identical magnitude per-pixel.
      self.assertAllClose(
          np.sum(x_, axis=3), np.sum(self._images, axis=3), atol=1e-3)

  def testInvalidShape(self):
    self.setUpWithNumOutputClasses(1)
    images = self._rng.randn(self._batch_size, self._height, self._width)
    with self.assertRaisesRegexp(snt.IncompatibleShapeError, "must have shape"):
      self._module(tf.convert_to_tensor(images))

  def testInvalidModelSize(self):
    self.setUpWithNumOutputClasses(1)
    module = snt.nets.Dilation(
        num_output_classes=self._num_output_classes,
        model_size="invalid_model_size")

    with self.assertRaisesRegexp(ValueError, "Unrecognized model_size"):
      module(tf.convert_to_tensor(self._images))

    # The other check for model_size being valid is only reached when
    # weight initializers are provided. We need to test this as well to get
    # 100% test coverage.
    module = snt.nets.Dilation(
        num_output_classes=self._num_output_classes,
        initializers={"w": snt.nets.noisy_identity_kernel_initializer(1)},
        model_size="invalid_model_size")
    with self.assertRaisesRegexp(ValueError, "Unrecognized model_size"):
      module(tf.convert_to_tensor(self._images))

  def test_properties(self):
    self.setUpWithNumOutputClasses(1)
    with self.assertRaises(snt.NotConnectedError):
      _ = self._module.conv_modules
    self._module(tf.convert_to_tensor(self._images))
    self.assertEqual(type(self._module.conv_modules), list)

  def testInvalidRegularizationParameters(self):
    regularizer = tf.contrib.layers.l1_regularizer(scale=0.5)
    with self.assertRaisesRegexp(KeyError, "Invalid regularizer keys.*"):
      self.setUpWithNumOutputClasses(1)
      snt.nets.Dilation(num_output_classes=self._num_output_classes,
                        regularizers={"not_w": regularizer})

    err = "Regularizer for 'w' is not a callable function"
    with self.assertRaisesRegexp(TypeError, err):
      self.setUpWithNumOutputClasses(1)
      snt.nets.Dilation(num_output_classes=self._num_output_classes,
                        regularizers={"w": tf.zeros([1, 2, 3])})

  def testRegularizersInRegularizationLosses(self):
    w_regularizer = tf.contrib.layers.l1_regularizer(scale=0.5)
    b_regularizer = tf.contrib.layers.l2_regularizer(scale=0.5)
    self.setUpWithNumOutputClasses(1)
    dilation_mod = snt.nets.Dilation(
        num_output_classes=self._num_output_classes,
        regularizers={"w": w_regularizer, "b": b_regularizer})
    dilation_mod(tf.convert_to_tensor(self._images))

    regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    # There are two regularizers per level
    layers_number = 8
    for i in range(0, 2 * layers_number, 2):
      self.assertRegexpMatches(regularizers[i].name, ".*l1_regularizer.*")
      self.assertRegexpMatches(regularizers[i + 1].name, ".*l2_regularizer.*")

  def testUtilities(self):
    err = "Cannot calculate range along non-existent index."
    with self.assertRaisesRegexp(ValueError, err):
      # Valid rank here would be either 0 or 1.
      dilation._range_along_dimension(2, [2, 4])


if __name__ == "__main__":
  tf.test.main()
