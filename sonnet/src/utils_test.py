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

"""Tests for sonnet.v2.src.utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
from sonnet.src import test_utils
from sonnet.src import utils
import tensorflow as tf


class ReplicateTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("Int", 42),
      ("Callable", lambda a: a))
  def testSingleValue(self, value):
    result = utils.replicate(value, 3, "value")
    self.assertLen(result, 3)
    self.assertAllEqual(result, (value,) * 3)

  @parameterized.named_parameters(
      ("Int", 42),
      ("String", "foo"),
      ("Callable", lambda a: a))
  def testListLengthOne(self, value):
    result = utils.replicate([value], 3, "value")
    self.assertLen(result, 3)
    self.assertAllEqual(result, (value,) * 3)

  @parameterized.named_parameters(
      ("Int", 42),
      ("String", "foo"),
      ("Callable", lambda a: a))
  def testTupleLengthN(self, value):
    v = (value,) * 3
    result = utils.replicate(v, 3, "value")
    self.assertLen(result, 3)
    self.assertAllEqual(result, (value,) * 3)

  @parameterized.named_parameters(
      ("Int", 42),
      ("String", "foo"),
      ("Callable", lambda a: a))
  def testListLengthN(self, value):
    v = list((value,) * 3)
    result = utils.replicate(v, 3, "value")
    self.assertLen(result, 3)
    self.assertAllEqual(result, (value,) * 3)

  def testIncorrectLength(self):
    v = [2, 2]
    with self.assertRaisesRegexp(
        TypeError,
        r"must be a scalar or sequence of length 1 or sequence of length 3"):
      utils.replicate(v, 3, "value")


class DecoratorTest(test_utils.TestCase):

  def test_callable_object(self):
    class MyObject(object):

      def __call__(self, x, y):
        return x ** y

    @utils.decorator
    def double(wrapped, instance, args, kwargs):
      self.assertIs(instance, o)
      return 2 * wrapped(*args, **kwargs)

    o = MyObject()
    f = double(o)  # pylint: disable=no-value-for-parameter
    self.assertEqual(f(3, y=4), 2 * (3 ** 4))

  def test_function(self):
    @utils.decorator
    def double(wrapped, instance, args, kwargs):
      self.assertIsNone(instance)
      return 2 * wrapped(*args, **kwargs)

    f = double(lambda x, y: x ** y)  # pylint: disable=no-value-for-parameter
    self.assertEqual(f(3, 4), 2 * (3 ** 4))

  def test_unbound_method(self):
    @utils.decorator
    def double(wrapped, instance, args, kwargs):
      self.assertIs(instance, o)
      return 2 * wrapped(*args, **kwargs)

    class MyObject(object):

      @double
      def f(self, x, y):
        return x ** y

    o = MyObject()
    self.assertEqual(o.f(3, 4), 2 * (3 ** 4))

  def test_bound_method(self):
    @utils.decorator
    def double(wrapped, instance, args, kwargs):
      self.assertIs(instance, o)
      return 2 * wrapped(*args, **kwargs)

    class MyObject(object):

      def f(self, x, y):
        return x ** y

    o = MyObject()
    self.assertEqual(double(o.f)(3, 4), 2 * (3 ** 4))  # pylint: disable=no-value-for-parameter


class ChannelIndexTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters("channels_first", "NCHW", "NC", "NCDHW")
  def test_returns_index_channels_first(self, data_format):
    self.assertEqual(utils.get_channel_index(data_format), 1)

  @parameterized.parameters("channels_last", "NHWC", "NDHWC", "BTWHD", "TBD")
  def test_returns_index_channels_last(self, data_format):
    self.assertEqual(utils.get_channel_index(data_format), -1)

  @parameterized.parameters("foo", "NCHC", "BTDTD", "chanels_first", "NHW")
  def test_invalid_strings(self, data_format):
    with self.assertRaisesRegexp(
        ValueError,
        "Unable to extract channel information from '{}'.".format(data_format)):
      utils.get_channel_index(data_format)


class AssertRankTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("tf_tensor", lambda rank: tf.ones([1] * rank)),
      ("tf_variable", lambda rank: tf.Variable(tf.ones([1] * rank))),
      ("tf_tensorspec", lambda rank: tf.TensorSpec([1] * rank)),
      ("np_ndarray", lambda rank: np.ones([1] * rank)))
  def test_valid_rank(self, input_fn):
    for rank in range(2, 5):
      inputs = input_fn(rank)
      utils.assert_rank(inputs, rank)
      utils.assert_minimum_rank(inputs, rank - 2)

  @parameterized.parameters(range(10))
  def test_invalid_rank(self, rank):
    x = tf.ones([1] * rank)
    # pylint: disable=g-error-prone-assert-raises
    with self.assertRaisesRegexp(ValueError, "must have rank %d" % (rank + 1)):
      utils.assert_rank(x, rank + 1)

    with self.assertRaisesRegexp(ValueError, "must have rank %d" % (rank - 1)):
      utils.assert_rank(x, rank - 1)

    with self.assertRaisesRegexp(ValueError,
                                 "must have rank >= %d" % (rank + 1)):
      utils.assert_minimum_rank(x, rank + 1)
    # pylint: enable=g-error-prone-assert-raises


class SmartAutographTest(test_utils.TestCase):

  def test_smart_ag(self):
    def foo(x):
      if x > 0:
        y = x * x
      else:
        y = -x
      return y

    with self.assertRaises(Exception):
      # Without autograph `foo` should not be traceable.
      func_foo = tf.function(foo, autograph=False)
      func_foo(tf.constant(2.))

    smart_foo = utils.smart_autograph(foo)
    func_smart_foo = tf.function(smart_foo, autograph=False)
    for x in tf.range(-10, 10):
      y = foo(x)
      self.assertAllEqual(smart_foo(x), y)
      self.assertAllEqual(func_smart_foo(x), y)

if __name__ == "__main__":
  # tf.enable_v2_behavior()
  tf.test.main()
