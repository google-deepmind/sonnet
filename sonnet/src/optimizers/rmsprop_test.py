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
"""Tests for sonnet.v2.src.rmsprop."""

from sonnet.src import test_utils
from sonnet.src.optimizers import optimizer_tests
from sonnet.src.optimizers import rmsprop
import tensorflow as tf

CONFIGS = optimizer_tests.named_product(learning_rate=(0.01, 0.001),
                                        decay=(0.8, 0.9),
                                        momentum=(0.0, 0.5),
                                        epsilon=(1e-7, 1e-8),
                                        centered=(False, True),
                                        seed=(28, 2, 90))


class ComparisonTest(optimizer_tests.AbstractFuzzTest):
  """Ensures Sonnet optimizers have equivalent behavior to TensorFlow."""

  def _make_tf(self, learning_rate, decay, momentum, epsilon, centered):
    optimizer = tf.optimizers.RMSprop(learning_rate=learning_rate,
                                      rho=decay,
                                      momentum=momentum,
                                      epsilon=epsilon,
                                      centered=centered)
    return lambda g, p: optimizer.apply_gradients(zip(g, p))

  def _make_snt(self, learning_rate, decay, momentum, epsilon, centered):
    optimizer = rmsprop.RMSProp(learning_rate=learning_rate,
                                decay=decay,
                                momentum=momentum,
                                epsilon=epsilon,
                                centered=centered)
    return optimizer.apply

  @test_utils.combined_named_parameters(CONFIGS)
  def testComparingSonnetAndTensorFlow(self, config):
    seed = config.pop("seed")
    self.assertParametersRemainClose(seed, config, atol=1e-2)


class RMSPropTest(optimizer_tests.OptimizerTestBase):

  def make_optimizer(self, **kwargs):
    if "learning_rate" not in kwargs:
      kwargs["learning_rate"] = 0.1
    return rmsprop.RMSProp(**kwargs)

  def testDense(self):
    parameters = [tf.Variable([1., 2.]), tf.Variable([3., 4.])]
    updates = [tf.constant([5., 5.]), tf.constant([3., 3.])]
    optimizer = self.make_optimizer(learning_rate=0.1)
    # Step 1 of RMSProp
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.683772, 1.683772], [2.683772, 3.683772]],
                        [x.numpy() for x in parameters])
    # Step 2 of RMSProp
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.454357, 1.454357], [2.454357, 3.454357]],
                        [x.numpy() for x in parameters])
    # Step 3 of RMSProp
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.262262, 1.262262], [2.262262, 3.262262]],
                        [x.numpy() for x in parameters])

  def testDenseCentered(self):
    parameters = [tf.Variable([1., 2.]), tf.Variable([3., 4.])]
    updates = [tf.constant([5., 5.]), tf.constant([3., 3.])]
    optimizer = self.make_optimizer(learning_rate=0.1, centered=True)
    # Step 1 of RMSProp
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.666667, 1.666667], [2.666667, 3.666667]],
                        [x.numpy() for x in parameters])
    # Step 2 of RMSProp
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.41176, 1.41176], [2.41176, 3.41176]],
                        [x.numpy() for x in parameters])
    # Step 3 of RMSProp
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.186776, 1.186776], [2.186776, 3.186776]],
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
    optimizer = self.make_optimizer(learning_rate=3.)
    # Step 1 of RMSProp
    optimizer.apply(updates, parameters)
    self.assertAllClose([[-8.486831], [2.0]], parameters[0].numpy(), rtol=1e-4)
    self.assertAllClose([[3.0], [-5.486784]], parameters[1].numpy(), rtol=1e-4)
    # Step 2 of RMSProp
    optimizer.apply(updates, parameters)
    self.assertAllClose([[-15.369301], [2.0]], parameters[0].numpy(), rtol=1e-4)
    self.assertAllClose([[3.0], [-12.369237]], parameters[1].numpy(), rtol=1e-4)
    # Step 3 of RMSProp
    optimizer.apply(updates, parameters)
    self.assertAllClose([[-21.132141], [2.0]], parameters[0].numpy(), rtol=1e-4)
    self.assertAllClose([[3.0], [-18.132067]], parameters[1].numpy(), rtol=1e-4)

  def testSparseCentered(self):
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
    optimizer = self.make_optimizer(learning_rate=3., centered=True)
    # Step 1 of RMSProp
    optimizer.apply(updates, parameters)
    self.assertAllClose([[-8.999999], [2.0]], parameters[0].numpy(), rtol=1e-4)
    self.assertAllClose([[3.0], [-5.999944]], parameters[1].numpy(), rtol=1e-4)
    # Step 2 of RMSProp
    optimizer.apply(updates, parameters)
    self.assertAllClose([[-16.64719], [2.0]], parameters[0].numpy(), rtol=1e-4)
    self.assertAllClose([[3.0], [-13.647109]], parameters[1].numpy(), rtol=1e-4)
    # Step 3 of RMSProp
    optimizer.apply(updates, parameters)
    self.assertAllClose([[-23.396709], [2.0]], parameters[0].numpy(), rtol=1e-4)
    self.assertAllClose([[3.0], [-20.39661]], parameters[1].numpy(), rtol=1e-4)

  def testVariableHyperParams(self):
    parameters = [tf.Variable([1., 2.]), tf.Variable([3., 4.])]
    updates = [tf.constant([5., 5.]), tf.constant([3., 3.])]
    learning_rate = tf.Variable(0.1)
    optimizer = self.make_optimizer(learning_rate=learning_rate)
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.683772, 1.683772], [2.683772, 3.683772]],
                        [x.numpy() for x in parameters])
    learning_rate.assign(0.01)
    self.assertAlmostEqual(0.01, optimizer.learning_rate.numpy())
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.660831, 1.660831], [2.660831, 3.660831]],
                        [x.numpy() for x in parameters])

  def testHyperParamDTypeConversion(self):
    parameters = [tf.Variable([1., 2.]), tf.Variable([3., 4.])]
    updates = [tf.constant([5., 5.]), tf.constant([3., 3.])]
    dtype = tf.float32 if self.primary_device == "TPU" else tf.float64
    learning_rate = tf.Variable(0.1, dtype=dtype)
    decay = tf.Variable(0.9, dtype=dtype)
    momentum = tf.Variable(0.0, dtype=dtype)
    epsilon = tf.Variable(1e-7, dtype=dtype)
    optimizer = self.make_optimizer(
        learning_rate=learning_rate,
        decay=decay,
        momentum=momentum,
        epsilon=epsilon)
    if optimizer_tests.is_tf_optimizer(optimizer):
      self.skipTest("TF optimizers don't support automatic casting.")

    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.683772, 1.683772], [2.683772, 3.683772]],
                        [x.numpy() for x in parameters])

  def testAuxVariablesColocatedWithOriginal(self):
    optimizer = self.make_optimizer(learning_rate=0.1)
    if optimizer_tests.is_tf_optimizer(optimizer):
      self.skipTest("Aux vars are in a different location for TF optimizers.")

    with tf.device("CPU:0"):
      var = tf.Variable(1.0)
    optimizer.apply([tf.constant(0.1)], [var])
    self.assertEqual(optimizer.mom[0].device, var.device)
    self.assertEqual(optimizer.ms[0].device, var.device)


class ReferenceRMSPropTest(RMSPropTest):

  def make_optimizer(self, **kwargs):
    if "learning_rate" not in kwargs:
      kwargs["learning_rate"] = 0.1
    kwargs["rho"] = kwargs.pop("decay", 0.9)
    if hasattr(tf.keras.optimizers, "legacy"):
      return optimizer_tests.WrappedTFOptimizer(
          tf.keras.optimizers.legacy.RMSprop(**kwargs))
    return optimizer_tests.WrappedTFOptimizer(
        tf.keras.optimizers.RMSprop(**kwargs))


if __name__ == "__main__":
  tf.test.main()
