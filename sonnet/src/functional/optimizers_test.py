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
"""Tests for functional optimizers."""

from absl.testing import parameterized
import sonnet as snt
from sonnet.src import test_utils
from sonnet.src.functional import haiku
from sonnet.src.functional import optimizers
import tensorflow as tf
import tree

sgd = optimizers.optimizer(snt.optimizers.SGD)
adam = optimizers.optimizer(snt.optimizers.Adam)


class OptimizersTest(test_utils.TestCase, parameterized.TestCase):

  def test_sgd(self):
    with haiku.variables():
      params = [tf.Variable(1.)]
      params = {p.ref(): tf.ones_like(p) for p in params}

    opt = sgd(learning_rate=0.01)
    opt_state = opt.init(params)
    grads = tree.map_structure(tf.ones_like, params)
    params, opt_state = opt.apply(opt_state, grads, params)
    p, = tree.flatten(params)
    self.assertAllClose(p.numpy(), 1. - (0.01 * 1))

  def test_adam(self):
    lin = haiku.transform(snt.Linear(1))
    x = tf.ones([1, 1])
    params = lin.init(x)

    optimizer = adam(learning_rate=0.01)
    opt_state = optimizer.init(params)
    # Step + (m, v) per parameter.
    self.assertLen(tree.flatten(opt_state), 5)

  @parameterized.parameters(True, False)
  def test_adam_with_variable_lr(self, trainable_lr):
    lin = haiku.transform(snt.Linear(1))
    x = tf.ones([1, 1])
    initial_params = lin.init(x)

    with haiku.variables():
      lr = tf.Variable(0.01, trainable=trainable_lr, name="lr")

    optimizer = adam(learning_rate=lr)
    initial_opt_state = optimizer.init(initial_params)
    # Learning rate, step + (m, v) per parameter.
    self.assertLen(tree.flatten(initial_opt_state), 6)

    grads = tree.map_structure(tf.ones_like, initial_params)
    params, opt_state = optimizer.apply(
        initial_opt_state, grads, initial_params)

    tree.assert_same_structure(initial_opt_state, opt_state)
    tree.assert_same_structure(initial_params, params)

if __name__ == "__main__":
  tf.test.main()
