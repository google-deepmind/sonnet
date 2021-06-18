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
"""Common tests for Sonnet optimizers."""

import itertools

from absl.testing import parameterized
import numpy as np
from sonnet.src import base
from sonnet.src import test_utils
import tensorflow as tf
import tree


class WrappedTFOptimizer(base.Optimizer):
  """Wraps a TF optimizer in the Sonnet API."""

  wrapped = None

  def __init__(self, optimizer: tf.optimizers.Optimizer):
    super().__init__()
    self.wrapped = optimizer

  def __getattr__(self, name):
    return getattr(self.wrapped, name)

  def apply(self, updates, params):
    self.wrapped.apply_gradients(zip(updates, params))


def is_tf_optimizer(optimizer):
  return isinstance(optimizer, WrappedTFOptimizer)


class OptimizerTestBase(test_utils.TestCase):
  """Common tests for Sonnet optimizers."""

  def make_optimizer(self, *args, **kwargs):
    raise NotImplementedError()

  def testNoneUpdate(self):
    parameters = [tf.Variable(1.), tf.Variable(2.)]
    updates = [None, tf.constant(3.)]
    optimizer = self.make_optimizer()
    optimizer.apply(updates, parameters)
    self.assertAllClose(1., parameters[0].numpy())

  def testDifferentLengthUpdatesParams(self):
    optimizer = self.make_optimizer()
    if is_tf_optimizer(optimizer):
      self.skipTest("TF optimizers don't check the lenghs of params/updates.")

    parameters = [tf.Variable([1., 2.]), tf.Variable([3., 4.])]
    updates = [tf.constant([5., 5.])]
    with self.assertRaisesRegex(
        ValueError, "`updates` and `parameters` must be the same length."):
      optimizer.apply(updates, parameters)

  def testEmptyParams(self):
    optimizer = self.make_optimizer()
    if is_tf_optimizer(optimizer):
      self.skipTest("TF optimizers don't error on empty params.")

    with self.assertRaisesRegex(ValueError, "`parameters` cannot be empty."):
      optimizer.apply([], [])

  def testAllUpdatesNone(self):
    parameters = [tf.Variable(1.), tf.Variable(2.)]
    updates = [None, None]
    optimizer = self.make_optimizer()
    if is_tf_optimizer(optimizer):
      msg = "No gradients provided for any variable"
    else:
      msg = "No updates provided for any parameter"
    with self.assertRaisesRegex(ValueError, msg):
      optimizer.apply(updates, parameters)

  def testInconsistentDTypes(self):
    optimizer = self.make_optimizer()
    if is_tf_optimizer(optimizer):
      self.skipTest("TF optimizers raise a cryptic error message here.")

    parameters = [tf.Variable([1., 2.], name="param0")]
    updates = [tf.constant([5, 5])]
    with self.assertRaisesRegex(
        ValueError, "DType of .* is not equal to that of parameter .*param0.*"):
      optimizer.apply(updates, parameters)

  def testUnsuppportedStrategyError(self):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
      parameters = [tf.Variable(1.0)]
      updates = [tf.constant(0.1)]
      optimizer = self.make_optimizer()
      if is_tf_optimizer(optimizer):
        self.skipTest("TF optimizers aren't restricted to Sonnet strategies.")
    with self.assertRaisesRegex(
        ValueError,
        "Sonnet optimizers are not compatible with `MirroredStrategy`"):
      strategy.run(lambda: optimizer.apply(updates, parameters))


# NOTE: Avoiding ABCMeta because of metaclass conflict.
class AbstractFuzzTest(test_utils.TestCase, parameterized.TestCase):
  """Tests TF and Sonnet run concurrently produce equivalent output."""

  def _make_tf(self, learning_rate, momentum, use_nesterov):
    raise NotImplementedError()

  def _make_snt(self, learning_rate, momentum, use_nesterov):
    raise NotImplementedError()

  def assertParametersRemainClose(self, seed, config, num_steps=100, atol=1e-4):
    tf_opt = self._make_tf(**config)
    snt_opt = self._make_snt(**config)

    # TODO(tomhennigan) Add sparse data.
    data = _generate_dense_data(seed, num_steps)
    tf_params = _apply_optimizer(data, tf_opt)
    snt_params = _apply_optimizer(data, snt_opt)
    assert tf_params and len(tf_params) == len(snt_params)

    for step, (tf_param, snt_param) in enumerate(zip(tf_params, snt_params)):
      msg = "TF and Sonnet diverged at step {}".format(step)
      for tf_p, snt_p in zip(tf_param, snt_param):
        self.assertEqual(tf_p.shape, snt_p.shape)
        self.assertEqual(tf_p.dtype, snt_p.dtype)
        self.assertAllClose(tf_p, snt_p, atol=atol, msg=msg)


def _generate_dense_data(seed, num_steps):
  """Generates deterministic random parameters and gradients."""
  # Use numpy random since it is deterministic (unlike TF).
  np.random.seed(seed=seed)
  params = [
      np.random.normal(size=(10, 10, 10)).astype(np.float32),
      np.random.normal(size=(10, 10)).astype(np.float32),
      np.random.normal(size=(10,)).astype(np.float32),
  ]
  per_step_grads = []
  for _ in range(num_steps):
    per_step_grads.append([
        np.random.normal(size=(10, 10, 10)).astype(np.float32),
        np.random.normal(size=(10, 10)).astype(np.float32),
        np.random.normal(size=(10,)).astype(np.float32),
    ])
  return params, per_step_grads


def _apply_optimizer(data, apply_fn):
  params, per_step_grads = data
  params = [tf.Variable(p, name="rank{}".format(len(p.shape))) for p in params]
  per_step_grads = tree.map_structure(tf.convert_to_tensor, per_step_grads)
  param_vals = []
  assert per_step_grads
  for grads in per_step_grads:
    apply_fn(grads, params)
    param_vals.append([p.numpy() for p in params])
  return param_vals


def named_product(**config):
  keys = list(config.keys())
  values = list(config.values())
  configs = []
  for val in itertools.product(*values):
    config = dict(zip(keys, val))
    name = ",".join("{}={}".format(k, v) for k, v in config.items())
    configs.append((name, config))
  return configs
