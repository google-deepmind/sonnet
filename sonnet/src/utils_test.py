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

from absl.testing import parameterized
import numpy as np
from sonnet.src import initializers
from sonnet.src import test_utils
from sonnet.src import utils
import tensorflow as tf

# We have a first "\" for the new line and one at the end. The rest is a direct
# copy-paste of the ground truth output.
_EXPECTED_FORMATTED_VARIABLE_LIST = ("""\
| Variable   | Spec     | Trainable   | Device   |
|------------+----------+-------------+----------|
| m1/v1      | f32[3,4] | True        | CPU      |
| m2/v2      | i32[5]   | False       | CPU      |\
""")


class ReplicateTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(("Int", 42), ("Callable", lambda a: a))
  def testSingleValue(self, value):
    result = utils.replicate(value, 3, "value")
    self.assertLen(result, 3)
    self.assertAllEqual(result, (value,) * 3)

  @parameterized.named_parameters(("Int", 42), ("String", "foo"),
                                  ("Callable", lambda a: a))
  def testListLengthOne(self, value):
    result = utils.replicate([value], 3, "value")
    self.assertLen(result, 3)
    self.assertAllEqual(result, (value,) * 3)

  @parameterized.named_parameters(("Int", 42), ("String", "foo"),
                                  ("Callable", lambda a: a))
  def testTupleLengthN(self, value):
    v = (value,) * 3
    result = utils.replicate(v, 3, "value")
    self.assertLen(result, 3)
    self.assertAllEqual(result, (value,) * 3)

  @parameterized.named_parameters(("Int", 42), ("String", "foo"),
                                  ("Callable", lambda a: a))
  def testListLengthN(self, value):
    v = list((value,) * 3)
    result = utils.replicate(v, 3, "value")
    self.assertLen(result, 3)
    self.assertAllEqual(result, (value,) * 3)

  def testIncorrectLength(self):
    v = [2, 2]
    with self.assertRaisesRegex(
        TypeError,
        r"must be a scalar or sequence of length 1 or sequence of length 3"):
      utils.replicate(v, 3, "value")


class DecoratorTest(test_utils.TestCase):

  def test_callable_object(self):

    class MyObject:

      def __call__(self, x, y):
        return x**y

    @utils.decorator
    def double(wrapped, instance, args, kwargs):
      self.assertIs(instance, o)
      return 2 * wrapped(*args, **kwargs)

    o = MyObject()
    f = double(o)  # pylint: disable=no-value-for-parameter
    self.assertEqual(f(3, y=4), 2 * (3**4))

  def test_function(self):

    @utils.decorator
    def double(wrapped, instance, args, kwargs):
      self.assertIsNone(instance)
      return 2 * wrapped(*args, **kwargs)

    f = double(lambda x, y: x**y)  # pylint: disable=no-value-for-parameter
    self.assertEqual(f(3, 4), 2 * (3**4))

  def test_unbound_method(self):

    @utils.decorator
    def double(wrapped, instance, args, kwargs):
      self.assertIs(instance, o)
      return 2 * wrapped(*args, **kwargs)

    class MyObject:

      @double
      def f(self, x, y):
        return x**y

    o = MyObject()
    self.assertEqual(o.f(3, 4), 2 * (3**4))

  def test_bound_method(self):

    @utils.decorator
    def double(wrapped, instance, args, kwargs):
      self.assertIs(instance, o)
      return 2 * wrapped(*args, **kwargs)

    class MyObject:

      def f(self, x, y):
        return x**y

    o = MyObject()
    self.assertEqual(double(o.f)(3, 4), 2 * (3**4))  # pylint: disable=no-value-for-parameter


class ChannelIndexTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters("channels_first", "NCHW", "NC", "NCDHW")
  def test_returns_index_channels_first(self, data_format):
    self.assertEqual(utils.get_channel_index(data_format), 1)

  @parameterized.parameters("channels_last", "NHWC", "NDHWC", "BTWHD", "TBD")
  def test_returns_index_channels_last(self, data_format):
    self.assertEqual(utils.get_channel_index(data_format), -1)

  @parameterized.parameters("foo", "NCHC", "BTDTD", "chanels_first", "NHW")
  def test_invalid_strings(self, data_format):
    with self.assertRaisesRegex(
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
    with self.assertRaisesRegex(ValueError, "must have rank %d" % (rank + 1)):
      utils.assert_rank(x, rank + 1)

    with self.assertRaisesRegex(ValueError, "must have rank %d" % (rank - 1)):
      utils.assert_rank(x, rank - 1)

    with self.assertRaisesRegex(ValueError,
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


class VariableLikeTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      [lambda: tf.constant([0., 1.]), lambda: tf.Variable([0., 1.])])
  def test_copies_shape(self, a):
    a = a()
    b = utils.variable_like(a)
    self.assertEqual(a.shape, b.shape)

  @parameterized.parameters([
      lambda: tf.constant(1, dtype=tf.int64),
      lambda: tf.Variable(1, dtype=tf.int64)
  ])
  def test_copies_dtype(self, a):
    a = a()
    b = utils.variable_like(a)
    self.assertEqual(a.dtype, b.dtype)

  @parameterized.parameters([lambda: tf.constant(1.), lambda: tf.Variable(1.)])
  def test_copies_device(self, a):
    with tf.device("CPU:0"):
      a = a()
    b = utils.variable_like(a)
    self.assertEqual(a.device, b.device)

  def test_default_initializer_is_zero(self):
    a = tf.Variable(1.)
    b = utils.variable_like(a)
    self.assertEqual(0., b.numpy())

  def test_override_initializer(self):
    a = tf.Variable(1.)
    b = utils.variable_like(a, initializer=initializers.Ones())
    self.assertEqual(1., b.numpy())

  @parameterized.parameters([True, False])
  def test_copies_variable_trainable(self, trainable):
    a = tf.Variable(1., trainable=trainable)
    b = utils.variable_like(a)
    self.assertEqual(a.trainable, b.trainable)

  def test_default_trainable_for_tensor(self):
    a = tf.constant(1.)
    b = utils.variable_like(a)
    self.assertEqual(True, b.trainable)

  @parameterized.parameters([True, False])
  def test_override_trainable(self, trainable):
    a = tf.Variable(1.)
    b = utils.variable_like(a, trainable=trainable)
    self.assertEqual(trainable, b.trainable)

  def test_copies_variable_name(self):
    a = tf.Variable(1., name="a")
    b = utils.variable_like(a)
    self.assertEqual(a.name, b.name)

  def test_default_name_for_tensor(self):
    a = tf.constant(1.)
    b = utils.variable_like(a)
    self.assertEqual("Variable:0", b.name)

  @parameterized.parameters([lambda: tf.constant(1.), lambda: tf.Variable(1.)])
  def test_override_name(self, a):
    a = a()
    b = utils.variable_like(a, name="b")
    self.assertEqual("b:0", b.name)


class FormatVariablesTest(test_utils.TestCase):

  def test_format_variables(self):
    with tf.device("/device:CPU:0"):
      with tf.name_scope("m1"):
        v1 = tf.Variable(tf.zeros([3, 4]), name="v1")
      with tf.name_scope("m2"):
        v2 = tf.Variable(
            tf.zeros([5], dtype=tf.int32), trainable=False, name="v2")
      self.assertEqual(
          utils.format_variables([v2, v1]), _EXPECTED_FORMATTED_VARIABLE_LIST)

  def test_log_variables(self):
    with tf.device("/device:CPU:0"):
      with tf.name_scope("m1"):
        v1 = tf.Variable(tf.zeros([3, 4]), name="v1")
      with tf.name_scope("m2"):
        v2 = tf.Variable(
            tf.zeros([5], dtype=tf.int32), trainable=False, name="v2")
      utils.log_variables([v2, v1])


class NotHashable:

  def __hash__(self):
    raise ValueError("Not hashable")


class CompareByIdTest(test_utils.TestCase):

  def test_access(self):
    original = NotHashable()
    wrapped = utils.CompareById(original)
    self.assertIs(wrapped.wrapped, original)

  def test_hash(self):
    original = NotHashable()
    wrapped = utils.CompareById(original)
    self.assertEqual(hash(wrapped), id(original))

  def test_eq(self):
    original1 = NotHashable()
    original2 = NotHashable()
    # Different wrappers pointing to the same object should be equal.
    self.assertEqual(utils.CompareById(original1), utils.CompareById(original1))
    # The original objet and the wrapped object should not be equal.
    self.assertNotEqual(original1, utils.CompareById(original1))
    # Similarly a different object should not be equal to a wrapped object.
    self.assertNotEqual(original2, utils.CompareById(original1))
    # None should also not compare.
    self.assertNotEqual(None, utils.CompareById(original1))
    # Different wrapped objects should not be equal.
    self.assertNotEqual(
        utils.CompareById(original1), utils.CompareById(original2))


if __name__ == "__main__":
  tf.test.main()
