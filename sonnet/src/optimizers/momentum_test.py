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
"""Tests for sonnet.v2.src.momentum."""

from sonnet.src import test_utils
from sonnet.src.optimizers import momentum as momentum_lib
from sonnet.src.optimizers import optimizer_tests
import tensorflow as tf

CONFIGS = optimizer_tests.named_product(learning_rate=(0.1, 0.01, 0.001),
                                        momentum=(0.9, 0.5, 0.2),
                                        use_nesterov=(True, False),
                                        seed=(28, 2, 90))


class ComparisonTest(optimizer_tests.AbstractFuzzTest):
  """Ensures Sonnet optimizers have equivalent behavior to TensorFlow."""

  def _make_tf(self, learning_rate, momentum, use_nesterov):
    optimizer = tf.optimizers.SGD(learning_rate=learning_rate,
                                  momentum=momentum,
                                  nesterov=use_nesterov)
    return lambda g, p: optimizer.apply_gradients(zip(g, p))

  def _make_snt(self, learning_rate, momentum, use_nesterov):
    optimizer = momentum_lib.Momentum(learning_rate=learning_rate,
                                      momentum=momentum,
                                      use_nesterov=use_nesterov)
    return optimizer.apply

  @test_utils.combined_named_parameters(CONFIGS)
  def testComparingSonnetAndTensorFlow(self, config):
    seed = config.pop("seed")
    self.assertParametersRemainClose(seed, config)


class MomentumTest(optimizer_tests.OptimizerTestBase):

  def make_optimizer(self, **kwargs):
    if "learning_rate" not in kwargs:
      kwargs["learning_rate"] = 0.1
    if "momentum" not in kwargs:
      kwargs["momentum"] = 0.9
    return momentum_lib.Momentum(**kwargs)

  def testDense(self):
    parameters = [tf.Variable([1., 2.]), tf.Variable([3., 4.])]
    updates = [tf.constant([5., 5.]), tf.constant([3., 3.])]
    optimizer = self.make_optimizer(learning_rate=0.1, momentum=0.9)
    # Step 1 of Momentum
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.5, 1.5], [2.7, 3.7]],
                        [x.numpy() for x in parameters])
    # Step 2 of Momentum
    optimizer.apply(updates, parameters)
    self.assertAllClose([[-0.45, 0.55], [2.13, 3.13]],
                        [x.numpy() for x in parameters])
    # Step 3 of Momentum
    optimizer.apply(updates, parameters)
    self.assertAllClose([[-1.805, -0.805], [1.317, 2.317]],
                        [x.numpy() for x in parameters])

  def testDenseNesterov(self):
    parameters = [tf.Variable([1., 2.]), tf.Variable([3., 4.])]
    updates = [tf.constant([5., 5.]), tf.constant([3., 3.])]
    optimizer = self.make_optimizer(
        learning_rate=0.1, momentum=0.9, use_nesterov=True)
    # Step 1 of Momentum
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.05, 1.05], [2.43, 3.43]],
                        [x.numpy() for x in parameters])
    # Step 2 of Momentum
    optimizer.apply(updates, parameters)
    self.assertAllClose([[-1.305, -0.305], [1.617, 2.617]],
                        [x.numpy() for x in parameters])
    # Step 3 of Momentum
    optimizer.apply(updates, parameters)
    self.assertAllClose([[-3.0245, -2.0245], [0.5853, 1.5853]],
                        [x.numpy() for x in parameters])

  def testSparse(self):
    if self.primary_device in ("GPU", "TPU"):
      self.skipTest("IndexedSlices not supported on {}.".format(
          self.primary_device))

    parameters = [tf.Variable([[1.], [2.]]), tf.Variable([[3.], [4.]])]
    updates = [
        tf.IndexedSlices(
            tf.constant([0.1], shape=[1, 1]), tf.constant([0]),
            tf.constant([2, 1])),
        tf.IndexedSlices(
            tf.constant([0.01], shape=[1, 1]), tf.constant([1]),
            tf.constant([2, 1]))
    ]
    optimizer = self.make_optimizer(learning_rate=3., momentum=0.9)
    # Step 1 of Momentum
    optimizer.apply(updates, parameters)
    self.assertAllClose([[1.0 - 3.0 * 0.1], [2.0]], parameters[0].numpy())
    self.assertAllClose([[3.0], [4.0 - 3.0 * 0.01]], parameters[1].numpy())
    # Step 2 of Momentum
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.7 - 3.0 * 0.19], [2.0]], parameters[0].numpy())
    self.assertAllClose([[3.0], [3.97 - 3.0 * 0.019]], parameters[1].numpy())
    # Step 3 of Momentum
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.13 - 3.0 * 0.271], [2.0]], parameters[0].numpy())
    self.assertAllClose([[3.0], [3.913 - 3.0 * 0.0271]], parameters[1].numpy())

  def testSparseNesterov(self):
    if self.primary_device in ("GPU", "TPU"):
      self.skipTest("IndexedSlices not supported on {}.".format(
          self.primary_device))

    parameters = [tf.Variable([[1.], [2.]]), tf.Variable([[3.], [4.]])]
    updates = [
        tf.IndexedSlices(
            tf.constant([0.1], shape=[1, 1]), tf.constant([0]),
            tf.constant([2, 1])),
        tf.IndexedSlices(
            tf.constant([0.01], shape=[1, 1]), tf.constant([1]),
            tf.constant([2, 1]))
    ]
    optimizer = self.make_optimizer(
        learning_rate=3., momentum=0.9, use_nesterov=True)
    # Step 1 of Momentum
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.43], [2.0]], parameters[0].numpy())
    self.assertAllClose([[3.0], [3.943]], parameters[1].numpy())
    # Step 2 of Momentum
    optimizer.apply(updates, parameters)
    self.assertAllClose([[-0.383], [2.0]], parameters[0].numpy())
    self.assertAllClose([[3.0], [3.8617]], parameters[1].numpy())
    # Step 3 of Momentum
    optimizer.apply(updates, parameters)
    self.assertAllClose([[-1.4147], [2.0]], parameters[0].numpy())
    self.assertAllClose([[3.0], [3.75853]], parameters[1].numpy())

  def testVariableHyperParams(self):
    parameters = [tf.Variable([1., 2.]), tf.Variable([3., 4.])]
    updates = [tf.constant([5., 5.]), tf.constant([3., 3.])]
    learning_rate = tf.Variable(0.1)
    momentum_coeff = tf.Variable(0.9)
    optimizer = self.make_optimizer(
        learning_rate=learning_rate, momentum=momentum_coeff)
    if optimizer_tests.is_tf_optimizer(optimizer):
      self.skipTest("TF SGD optimizer doesn't support variable learning rate.")

    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.5, 1.5], [2.7, 3.7]],
                        [x.numpy() for x in parameters])
    learning_rate.assign(0.01)
    momentum_coeff.assign(0.09)
    self.assertAlmostEqual(0.01, optimizer.learning_rate.numpy())
    self.assertAlmostEqual(0.09, optimizer.momentum.numpy())
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.4455, 1.4455], [2.6673, 3.6673]],
                        [x.numpy() for x in parameters])

  def testHyperParamDTypeConversion(self):
    parameters = [tf.Variable([1., 2.]), tf.Variable([3., 4.])]
    updates = [tf.constant([5., 5.]), tf.constant([3., 3.])]
    dtype = tf.float32 if self.primary_device == "TPU" else tf.float64
    learning_rate = tf.Variable(0.1, dtype=dtype)
    momentum_coeff = tf.Variable(0.9, dtype=dtype)
    optimizer = self.make_optimizer(
        learning_rate=learning_rate, momentum=momentum_coeff)
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.5, 1.5], [2.7, 3.7]],
                        [x.numpy() for x in parameters])

  def testAuxVariablesColocatedWithOriginal(self):
    optimizer = self.make_optimizer(learning_rate=0.1, momentum=0.9)
    if optimizer_tests.is_tf_optimizer(optimizer):
      self.skipTest("TF slot variables are in a different location.")

    with tf.device("CPU:0"):
      var = tf.Variable(1.0)
    optimizer.apply([tf.constant(0.1)], [var])
    self.assertEqual(optimizer.accumulated_momentum[0].device, var.device)


class ReferenceMomentumTest(MomentumTest):

  def make_optimizer(self, **kwargs):
    if "learning_rate" not in kwargs:
      kwargs["learning_rate"] = 0.1
    if "momentum" not in kwargs:
      kwargs["momentum"] = 0.9
    if "use_nesterov" in kwargs:
      kwargs["nesterov"] = kwargs["use_nesterov"]
      del kwargs["use_nesterov"]
    if hasattr(tf.keras.optimizers, "legacy"):
      return optimizer_tests.WrappedTFOptimizer(
          tf.keras.optimizers.legacy.SGD(**kwargs))
    return optimizer_tests.WrappedTFOptimizer(tf.keras.optimizers.SGD(**kwargs))


if __name__ == "__main__":
  tf.test.main()
