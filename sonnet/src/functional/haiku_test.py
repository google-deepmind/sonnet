# Copyright 2020 The Sonnet Authors. All Rights Reserved.
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
"""Tests for Haiku compatibility layer."""

from absl.testing import parameterized
import sonnet as snt
from sonnet.src import test_utils
from sonnet.src.functional import haiku as hk
import tensorflow as tf
import tree


class TensorVariableTest(test_utils.TestCase, parameterized.TestCase):

  def test_initial_value(self):
    with hk.variables():
      v = tf.Variable(tf.ones([]))
    self.assertIsInstance(v, hk.TensorVariable)
    self.assertAllEqual(v, 1)
    self.assertAllEqual(v.read_value(), 1)
    self.assertAllEqual(v.tensor_value, 1)

  @parameterized.parameters(None, True, False)
  def test_trainable(self, trainable):
    with hk.variables():
      v = tf.Variable(1., trainable=trainable)
    if trainable is None:
      self.assertTrue(v.trainable)
    else:
      self.assertEqual(v.trainable, trainable)

  def test_name(self):
    with hk.variables():
      v = tf.Variable(tf.ones([]), name="v")
    self.assertEqual(v.name, "v:0")

  def test_name_with_scope(self):
    with hk.variables(), tf.name_scope("foo"), tf.name_scope("bar"):
      v = tf.Variable(tf.ones([]), name="v")
    self.assertEqual(v.name, "foo/bar/v:0")

  @parameterized.parameters(([],), ([1, 2, 3],))
  def test_shape(self, shape):
    with hk.variables():
      v = tf.Variable(tf.ones(shape))
    self.assertEqual(shape, v.shape.as_list())

  @parameterized.parameters(tf.float32, tf.int32)
  def test_dtype(self, dtype):
    with hk.variables():
      v = tf.Variable(tf.ones([], dtype=dtype))
    self.assertEqual(dtype, v.dtype)

  def test_attributes_do_not_notify(self):
    with hk.variables():
      v = tf.Variable(1.)
      s = tf.Variable(1., trainable=False)

    def f():
      for c in (v, s):
        self.assertIsNotNone(c.shape)
        self.assertIsNotNone(c.dtype)
        self.assertIsNotNone(c.trainable)
        self.assertIsNotNone(c.name)
        self.assertIsNotNone(c.device)

    f = hk.transform_with_state(f)
    params, state = f.init()
    self.assertEmpty(params)
    self.assertEmpty(state)

    out, state = f.apply(params, state)
    self.assertIsNone(out)
    self.assertEmpty(state)

  def test_read_captured_variables_included(self):
    with hk.variables():
      v = tf.Variable(1.)
      s = tf.Variable(1., trainable=False)

    f = hk.transform_with_state(lambda: (v.read_value() + s.read_value()))

    params, state = f.init()
    self.assertEqual(params, {v.ref(): v.tensor_value})
    self.assertEqual(state, {s.ref(): s.tensor_value})

  def test_captured_variable_from_other_function_raises(self):
    def f(model):
      if not model:
        model.append(tf.Variable(1.))
        model.append(tf.Variable(1., trainable=False))
      return sum(model)

    f = hk.transform_with_state(f)

    model = []
    params, state = f.init(model)
    self.assertLen(params, 1)
    self.assertLen(state, 1)

    with self.assertRaisesRegex(ValueError, "TensorVariable .* has no value"):
      f.init(model)

  def test_assign(self):
    with hk.variables():
      v = tf.Variable(tf.ones([]))
    v.assign(tf.zeros([]))
    self.assertAllEqual(v.numpy(), 0)
    self.assertAllEqual(v.read_value().numpy(), 0)
    self.assertAllEqual(v.tensor_value.numpy(), 0)

  def test_assign_add(self):
    with hk.variables():
      v = tf.Variable(tf.ones([]))
    v.assign_add(1.)
    self.assertAllEqual(v.numpy(), 2)
    self.assertAllEqual(v.read_value().numpy(), 2)
    self.assertAllEqual(v.tensor_value.numpy(), 2)

  def test_assign_sub(self):
    with hk.variables():
      v = tf.Variable(tf.ones([]))
    v.assign_sub(1.)
    self.assertAllEqual(v.numpy(), 0)
    self.assertAllEqual(v.read_value().numpy(), 0)
    self.assertAllEqual(v.tensor_value.numpy(), 0)


class NetworkTest(test_utils.TestCase, parameterized.TestCase):

  def test_transform(self):
    mod = snt.Linear(1, w_init=tf.ones)
    snt.allow_empty_variables(mod)
    self.assertEmpty(mod.variables)

    f = hk.transform(mod)
    x = tf.ones([1, 1])

    params = f.init(x)
    self.assertLen(params.items(), 2)
    self.assertAllEqual(params[mod.w.ref()], [[1.]])
    self.assertAllEqual(params[mod.b.ref()], [0.])

    y = f.apply(params, x)
    self.assertEqual(y, [[1.]])

    params = tree.map_structure(lambda p: p + 1, params)
    y = f.apply(params, x)
    self.assertEqual(y, [[3.]])

  def test_initial_values_preserved(self):
    with hk.variables():
      v = tf.Variable(0)
      v.assign(1)

    def assert_values():
      self.assertEqual(v.initial_tensor_value.numpy(), 0)
      self.assertEqual(v.tensor_value.numpy(), 1)

    assert_values()
    f = hk.transform(lambda: v.assign(2))
    assert_values()
    params = f.init()
    assert_values()
    f.apply(params)
    assert_values()

  def test_variables_in_transform_set_to_none(self):
    mod = snt.Bias()
    f = hk.transform(mod)
    params = f.init(tf.ones([1, 1]))  # Will create `mod.b`.
    self.assertIsNone(mod.b.tensor_value)
    self.assertIsNone(mod.b.initial_tensor_value)

    y = f.apply(params, tf.ones([1, 1]))
    self.assertAllEqual(y.numpy(), [[1.]])
    self.assertIsNone(mod.b.tensor_value)
    self.assertIsNone(mod.b.initial_tensor_value)

  def test_disallows_variables_in_apply(self):
    _, apply_fn = hk.transform(lambda: tf.Variable(1))
    with self.assertRaisesRegex(ValueError,
                                "Apply function cannot create new variables"):
      apply_fn({})

  def test_state_returns_initial_value(self):
    with hk.variables():
      # NOTE: Initial value defined outside transform.
      v = tf.Variable(0, trainable=False)

    f = hk.transform_with_state(lambda: v.assign(1))
    params, state = f.init()
    initial_v = state[v.ref()]
    self.assertEqual(initial_v.numpy(), 0)

    y, state = f.apply(params, state)
    final_v = state[v.ref()]
    self.assertEqual(y.numpy(), 1)
    self.assertEqual(final_v.numpy(), 1)

  def test_state_counter(self):
    with hk.variables():
      v = tf.Variable(0, trainable=False)

    f = hk.transform_with_state(lambda: v.assign_add(1))
    params, initial_state = f.init()
    for _ in range(2):
      state = initial_state
      for i in range(10):
        y, state = f.apply(params, state)
        self.assertEqual(y.numpy(), i + 1)

  def test_state_ema(self):
    with hk.variables():
      ema = snt.ExponentialMovingAverage(decay=0.5)
    ema = hk.transform_with_state(ema)

    params, state = ema.init(3.0)
    y, state = ema.apply(params, state, 3.0)
    self.assertAllClose(y.numpy(), 3.0)
    y, state = ema.apply(params, state, 6.0)
    self.assertAllClose(y.numpy(), 5.0)

if __name__ == "__main__":
  tf.test.main()
