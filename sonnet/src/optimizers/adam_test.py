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
"""Tests for sonnet.v2.src.adam."""

from sonnet.src import test_utils
from sonnet.src.optimizers import adam
from sonnet.src.optimizers import optimizer_tests
import tensorflow as tf

CONFIGS = optimizer_tests.named_product(learning_rate=(0.1, 0.01, 0.001),
                                        beta_1=(0.9, 0.99, 0.999),
                                        beta_2=(0.9, 0.99, 0.999),
                                        epsilon=(1e-8,),
                                        seed=(28, 2, 90))


class ComparisonTest(optimizer_tests.AbstractFuzzTest):
  """Ensures Sonnet optimizers have equivalent behavior to TensorFlow."""

  def _make_tf(self, learning_rate, beta_1, beta_2, epsilon):
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate,
                                   beta_1=beta_1,
                                   beta_2=beta_2,
                                   epsilon=epsilon)
    return lambda g, p: optimizer.apply_gradients(zip(g, p))

  def _make_snt(self, learning_rate, beta_1, beta_2, epsilon):
    optimizer = adam.Adam(learning_rate=learning_rate,
                          beta1=beta_1,
                          beta2=beta_2,
                          epsilon=epsilon)
    return optimizer.apply

  @test_utils.combined_named_parameters(CONFIGS)
  def testComparingSonnetAndTensorFlow(self, config):
    seed = config.pop("seed")
    self.assertParametersRemainClose(seed, config)


class AdamTest(optimizer_tests.OptimizerTestBase):

  def make_optimizer(self, **kwargs):
    if "learning_rate" not in kwargs:
      kwargs["learning_rate"] = 0.001
    return adam.Adam(**kwargs)

  def testDense(self):
    parameters = [tf.Variable([1., 2.]), tf.Variable([3., 4.])]
    updates = [tf.constant([5., 5.]), tf.constant([3., 3.])]
    optimizer = self.make_optimizer(learning_rate=0.001)
    # Step 1 of Adam
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.999, 1.999], [2.999, 3.999]],
                        [x.numpy() for x in parameters])
    # Step 2 of Adam
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.998, 1.998], [2.998, 3.998]],
                        [x.numpy() for x in parameters])
    # Step 3 of Adam
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.997, 1.997], [2.997, 3.997]],
                        [x.numpy() for x in parameters])

  def testSparse(self):
    if self.primary_device in ("GPU", "TPU"):
      self.skipTest("IndexedSlices not supported on {}.".format(
          self.primary_device))

    parameters = [tf.Variable([[1.], [2.]]), tf.Variable([[3.], [4.]])]
    tf_parameters = [tf.Variable([[1.], [2.]]), tf.Variable([[3.], [4.]])]
    updates = [
        tf.IndexedSlices(
            tf.constant([0.1], shape=[1, 1]), tf.constant([0]),
            tf.constant([2, 1])),
        tf.IndexedSlices(
            tf.constant([0.01], shape=[1, 1]), tf.constant([1]),
            tf.constant([2, 1]))
    ]
    optimizer = self.make_optimizer(learning_rate=0.001)

    # Compare against TF optimizer.
    tf_optimizer = tf.optimizers.Adam(learning_rate=0.001)
    # Step 1 of Adam
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.999], [2.0]], parameters[0].numpy())
    self.assertAllClose([[3.0], [3.999]], parameters[1].numpy())
    tf_optimizer.apply_gradients(zip(updates, tf_parameters))
    self.assertAllClose(tf_parameters[0].numpy(), parameters[0].numpy())
    self.assertAllClose(tf_parameters[1].numpy(), parameters[1].numpy())
    # Step 2 of Adam
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.998], [2.0]], parameters[0].numpy())
    self.assertAllClose([[3.0], [3.998]], parameters[1].numpy())
    tf_optimizer.apply_gradients(zip(updates, tf_parameters))
    self.assertAllClose(tf_parameters[0].numpy(), parameters[0].numpy())
    self.assertAllClose(tf_parameters[1].numpy(), parameters[1].numpy())
    # Step 3 of Adam
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.997], [2.0]], parameters[0].numpy())
    self.assertAllClose([[3.0], [3.997]], parameters[1].numpy())
    tf_optimizer.apply_gradients(zip(updates, tf_parameters))
    self.assertAllClose(tf_parameters[0].numpy(), parameters[0].numpy())
    self.assertAllClose(tf_parameters[1].numpy(), parameters[1].numpy())

  def testVariableHyperParams(self):
    parameters = [tf.Variable([1., 2.]), tf.Variable([3., 4.])]
    updates = [tf.constant([5., 5.]), tf.constant([3., 3.])]
    learning_rate = tf.Variable(0.001)
    optimizer = self.make_optimizer(learning_rate=learning_rate)
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.999, 1.999], [2.999, 3.999]],
                        [x.numpy() for x in parameters])
    learning_rate.assign(0.1)
    self.assertAlmostEqual(0.1, optimizer.learning_rate.numpy())
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.899, 1.899], [2.899, 3.899]],
                        [x.numpy() for x in parameters],
                        rtol=1e-4)

  def testHyperParamDTypeConversion(self):
    parameters = [tf.Variable([1., 2.]), tf.Variable([3., 4.])]
    updates = [tf.constant([5., 5.]), tf.constant([3., 3.])]
    dtype = tf.float32 if self.primary_device == "TPU" else tf.float64
    learning_rate = tf.Variable(0.001, dtype=dtype)
    beta1 = tf.Variable(0.9, dtype=dtype)
    beta2 = tf.Variable(0.999, dtype=dtype)
    epsilon = tf.Variable(1e-8, dtype=dtype)
    optimizer = self.make_optimizer(
        learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
    optimizer.apply(updates, parameters)
    self.assertAllClose([[0.999, 1.999], [2.999, 3.999]],
                        [x.numpy() for x in parameters],
                        rtol=1e-4)

  def testAuxVariablesColocatedWithOriginal(self):
    optimizer = self.make_optimizer(learning_rate=0.001)
    with tf.device("CPU:0"):
      var = tf.Variable(1.0)
    optimizer.apply([tf.constant(0.1)], [var])
    self.assertEqual(optimizer.m[0].device, var.device)
    self.assertEqual(optimizer.v[0].device, var.device)


class ReferenceAdamTest(optimizer_tests.OptimizerTestBase):

  def make_optimizer(self, **kwargs):
    if "learning_rate" not in kwargs:
      kwargs["learning_rate"] = 0.001
    return optimizer_tests.WrappedTFOptimizer(tf.optimizers.Adam(**kwargs))


if __name__ == "__main__":
  tf.test.main()
