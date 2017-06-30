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

"""Tests for sonnet.python.modules.batch_norm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import sonnet as snt
from sonnet.testing import parameterized
import tensorflow as tf

from tensorflow.python.ops import variables


class BatchNormTest(parameterized.ParameterizedTestCase, tf.test.TestCase):

  def testConstruct(self):
    inputs = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])

    batch_norm1 = snt.BatchNorm(offset=False, scale=False)
    batch_norm1(inputs, is_training=True)

    err = "Batch normalization doesn't have an offset, so no beta"
    with self.assertRaisesRegexp(snt.Error, err):
      _ = batch_norm1.beta

    err = "Batch normalization doesn't have a scale, so no gamma"
    with self.assertRaisesRegexp(snt.Error, err):
      _ = batch_norm1.gamma

    batch_norm2 = snt.BatchNorm(offset=True, scale=False)
    batch_norm2(inputs, is_training=True)
    _ = batch_norm2.beta

    batch_norm3 = snt.BatchNorm(offset=False, scale=True)
    batch_norm3(inputs, is_training=True)
    _ = batch_norm3.gamma

    batch_norm4 = snt.BatchNorm(offset=True, scale=True)
    batch_norm4(inputs, is_training=True)
    _ = batch_norm4.beta
    _ = batch_norm4.gamma

    batch_norm4(inputs, is_training=True, test_local_stats=True)
    batch_norm4(inputs,
                is_training=tf.constant(True),
                test_local_stats=tf.constant(True))

    is_training_ph = tf.placeholder(tf.bool)
    test_local_stats_ph = tf.placeholder(tf.bool)
    batch_norm4(inputs,
                is_training=is_training_ph,
                test_local_stats=test_local_stats_ph)

  def testReductionIndices(self):
    """Check that differing reduction indices give the correct output shape."""

    inputs = tf.placeholder(tf.float32, shape=[None, 64, 32, 3])

    bn1 = snt.BatchNorm(axis=[0], offset=False)
    bn1(inputs, is_training=True)
    self.assertEqual(bn1.moving_mean.get_shape(), (1, 64, 32, 3))

    bn2 = snt.BatchNorm(axis=[0, 1], offset=False)
    bn2(inputs, is_training=True)
    self.assertEqual(bn2.moving_mean.get_shape(), (1, 1, 32, 3))

    bn3 = snt.BatchNorm(axis=[0, 2], offset=False)
    bn3(inputs, is_training=True)
    self.assertEqual(bn3.moving_mean.get_shape(), (1, 64, 1, 3))

    bn4 = snt.BatchNorm(offset=False)
    bn4(inputs, is_training=True)
    self.assertEqual(bn4.moving_mean.get_shape(), (1, 1, 1, 3))

    err = (r"Too many indices specified in axis: "
           r"len\(\[0, 1, 2, 3, 0\]\) > len\(\(\?, 64, 32, 3\)\)")
    with self.assertRaisesRegexp(snt.IncompatibleShapeError, err):
      bn5 = snt.BatchNorm(axis=[0, 1, 2, 3, 0])
      bn5(inputs, is_training=True)

    err = r"One or more index in axis is too large for input shape: \[4\] >= 4"
    with self.assertRaisesRegexp(snt.IncompatibleShapeError, err):
      bn6 = snt.BatchNorm(axis=[4])
      bn6(inputs, is_training=True)

    err = r"Indices in axis must be non-negative: \[-1\] < 0"
    with self.assertRaisesRegexp(snt.IncompatibleShapeError, err):
      bn7 = snt.BatchNorm(axis=[-1])
      bn7(inputs, is_training=True)

  def testFloat16Error(self):
    inputs = tf.placeholder(tf.float16, shape=[None, 64, 32, 3])
    batch_norm = snt.BatchNorm()

    err = (r"BatchNorm does not support `tf\.float16`, insufficient precision "
           "for calculating sufficient statistics.")
    with self.assertRaisesRegexp(snt.NotSupportedError, err):
      batch_norm(inputs, is_training=True)

  @parameterized.NamedParameters(
      ("Float32", tf.float32),
      ("Float64", tf.float64),
  )
  def testDataType(self, dtype):
    inputs = tf.placeholder(dtype, shape=[None, 64, 32, 3])
    batch_norm = snt.BatchNorm(offset=True, scale=True)
    output = batch_norm(inputs, is_training=True)

    self.assertEqual(dtype, output.dtype)
    self.assertEqual(dtype, batch_norm.moving_mean.dtype.base_dtype)
    self.assertEqual(dtype, batch_norm.moving_variance.dtype.base_dtype)
    self.assertEqual(dtype, batch_norm.gamma.dtype.base_dtype)
    self.assertEqual(dtype, batch_norm.beta.dtype.base_dtype)

  def _get_inputs(self, dtype=tf.float32):
    v = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype.as_numpy_dtype)
    input_v = np.array([v, v, v, v, v, v, v])
    inputs = tf.constant(input_v)

    return v, input_v, inputs

  def testShiftImproveStatistics(self):
    """Test that using moving_mean as shift improves statistics."""

    _, _, inputs = self._get_inputs()

    # Use small decay_rate to update faster.
    bn = snt.BatchNorm(offset=False, scale=False, decay_rate=0.1)
    out1 = bn(inputs, is_training=True)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      out_v = sess.run(out1)

      # Before updating the moving_mean the results are off.
      self.assertAllClose(np.zeros([7, 6]), out_v, rtol=1e-6, atol=1e-5)

      sess.run(tuple(tf.get_collection(tf.GraphKeys.UPDATE_OPS)))

      # After updating the moving_mean the results are better.
      out_v = sess.run(out1)
      self.assertAllClose(np.zeros([7, 6]), out_v, rtol=1e-6, atol=1e-6)

  @parameterized.NamedParameters(
      ("Float32", tf.float32),
      ("Float64", tf.float64),
  )
  def testCheckStatsDouble(self, dtype):
    """The correct statistics are being computed for double connection.

    Connected in parallel, it's ill-defined what order the updates will happen
    in. A double update could happen, or two sequential updates. E.g. If
    decay_rate is 0.9, the start value is 1.0, and the target value is 0.0, the
    value could progress as

      1.00 -> 0.90 -> 0.81,

    if the second update uses the fresh second value. Or as

      1.00 -> 0.90 -> 0.80

    if the second update uses the stale first value.

    We fix this here by running them in sequential run calls to ensure that this
    test is deterministic.

    The two situations are minimally different, especially if decay_rate is
    close to one (e.g. the default of 0.999).

    Args:
      dtype: TensorFlow datatype of input test batch.
    """

    v, _, inputs = self._get_inputs(dtype)
    bn = snt.BatchNorm(offset=False, scale=False, decay_rate=0.9)

    with tf.name_scope("net1"):
      bn(inputs, is_training=True)

    with tf.name_scope("net2"):
      bn(inputs, is_training=True)

    update_ops_1 = tuple(tf.get_collection(tf.GraphKeys.UPDATE_OPS, "net1"))
    self.assertEqual(len(update_ops_1), 2)
    update_ops_2 = tuple(tf.get_collection(tf.GraphKeys.UPDATE_OPS, "net2"))
    self.assertEqual(len(update_ops_2), 2)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())

      mm, mv = sess.run([bn.moving_mean, bn.moving_variance])

      self.assertAllClose(np.zeros([1, 6]), mm)
      self.assertAllClose(np.ones([1, 6]), mv)

      sess.run(update_ops_1)
      sess.run(update_ops_2)

      mm, mv = sess.run([bn.moving_mean,
                         bn.moving_variance])

      correct_mm = (1.0 - bn._decay_rate) * v
      correct_mm = (1.0 - bn._decay_rate) * v + bn._decay_rate * correct_mm
      correct_mv = np.ones([1, 6]) * bn._decay_rate**2

      self.assertAllClose(np.reshape(correct_mm, [1, 6]), mm)
      self.assertAllClose(np.reshape(correct_mv, [1, 6]), mv)

  def testCheckStatsPython(self):
    """The correct normalization is being used for different Python flags."""

    v, input_v, inputs = self._get_inputs()

    bn = snt.BatchNorm(offset=False, scale=False, decay_rate=0.5)
    out1 = bn(inputs, is_training=True, test_local_stats=True)
    out2 = bn(inputs, is_training=False, test_local_stats=True)
    out3 = bn(inputs, is_training=False, test_local_stats=False)

    update_ops = tuple(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    self.assertEqual(len(update_ops), 2)

    with tf.control_dependencies(update_ops):
      out1 = tf.identity(out1)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())

      out_v = sess.run(out1)
      mm, mv = sess.run([bn.moving_mean, bn.moving_variance])

      # Single moving average steps should have happened.
      correct_mm = (1.0 - bn._decay_rate) * v
      correct_mv = np.ones([1, 6]) * bn._decay_rate

      self.assertAllClose(np.reshape(correct_mm, [1, 6]), mm)
      self.assertAllClose(np.reshape(correct_mv, [1, 6]), mv)
      self.assertAllClose(np.zeros([7, 6]), out_v, rtol=1e-6, atol=1e-5)

      out2_, out3_ = sess.run([out2, out3])

      # Out2: Tested using local batch stats.
      # Better numerical precision due to using shifted estimators.
      self.assertAllClose(np.zeros([7, 6]), out2_)

      # Out3: Tested using moving average stats.
      self.assertAllClose(
          (input_v - mm) / np.sqrt(mv + bn._eps),
          out3_)

  @parameterized.NamedParameters(
      ("UseUpdateCollection", tf.GraphKeys.UPDATE_OPS),
      ("UseDifferentUpdateCollection", "my_update_ops"),
      ("UseControlDependencies", None),
  )
  def testCheckStatsInGraph(self, update_ops_collection):
    """The correct normalization is being used for different TF flags."""

    v, input_v, inputs = self._get_inputs()

    bn = snt.BatchNorm(offset=False,
                       scale=False,
                       decay_rate=0.5,
                       update_ops_collection=update_ops_collection)

    is_training = tf.placeholder(tf.bool)
    test_local_stats = tf.placeholder(tf.bool)

    out = bn(inputs,
             is_training=is_training,
             test_local_stats=test_local_stats)

    if update_ops_collection is not None:
      update_ops = tuple(tf.get_collection(update_ops_collection))
      self.assertEqual(len(update_ops), 2)

      with tf.control_dependencies(update_ops):
        out = tf.identity(out)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())

      # Run with `is_training=True`, `test_local_stats=True`.
      out_v = sess.run(out, feed_dict={is_training: True,
                                       test_local_stats: True})

      # Moving averages not updated until after calculation so shifted
      # stats are poor.
      self.assertAllClose(np.zeros([7, 6]), out_v, rtol=1e-6, atol=1e-5)

      ops = (bn.moving_mean, bn.moving_variance)
      mm1, mv1 = sess.run(ops)

      # Single moving average step should have happened.
      correct_mm = (1.0 - bn._decay_rate) * v
      correct_mv = np.ones([1, 6]) * bn._decay_rate

      self.assertAllClose(np.reshape(correct_mm, [1, 6]), mm1)
      self.assertAllClose(np.reshape(correct_mv, [1, 6]), mv1)

      # Run with `is_training=False`, `test_local_stats=True`.
      # Should have used local batch stats.
      out_v = sess.run(out, feed_dict={is_training: False,
                                       test_local_stats: True})

      # Moving averages should not have changed.
      mm2, mv2 = sess.run(ops)
      self.assertAllClose(mm1, mm2)
      self.assertAllClose(mv1, mv2)

      self.assertAllClose(np.zeros([7, 6]), out_v)

      # Run with `is_training=False`, `test_local_stats=False`.
      # Should have used moving average stats.
      out_v = sess.run(out, feed_dict={is_training: False,
                                       test_local_stats: False})

      # Moving averages should not have changed.
      mm3, mv3 = sess.run(ops)
      self.assertAllClose(mm1, mm3)
      self.assertAllClose(mv1, mv3)

      self.assertAllClose(
          (input_v - mm3) / np.sqrt(mv3 + bn._eps),
          out_v)

  def testSharing(self):
    """Check that the correct number of variables are made when sharing."""

    inputs1 = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
    inputs2 = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])

    bn = snt.BatchNorm(offset=True, scale=True)

    bn(inputs1, is_training=True)
    bn(inputs2, is_training=False)

    self.assertEqual(len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)), 4)

    # We should have one set of update ops
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    self.assertEqual(len(update_ops), 2)

  def testUpdatesInsideCond(self):
    """Demonstrate that updates inside a cond fail.

    """

    _, input_v, inputs = self._get_inputs()
    bn = snt.BatchNorm(offset=False, scale=False, decay_rate=0.5)
    condition = tf.placeholder(tf.bool)
    cond = tf.cond(condition,
                   lambda: bn(inputs, is_training=True),
                   lambda: inputs)

    init = tf.global_variables_initializer()

    with self.test_session() as sess:
      sess.run(init)
      out_v = sess.run(cond, feed_dict={condition: False})
      self.assertAllClose(input_v, out_v)

      out_v = sess.run(cond, feed_dict={condition: True})
      self.assertAllClose(np.zeros([7, 6]), out_v, rtol=1e-4, atol=1e-4)

      # Variables are accessible outside the tf.cond()
      mm, mv = sess.run([bn.moving_mean, bn.moving_variance])
      self.assertAllClose(np.zeros([1, 6]), mm)
      self.assertAllClose(np.ones([1, 6]), mv)

      # Tensors are not accessible outside the tf.cond()
      with self.assertRaisesRegexp(ValueError, "Operation"):
        sess.run(tuple(tf.get_collection(tf.GraphKeys.UPDATE_OPS)))

  def testVariableBatchSize(self):
    """Check the inputs batch_size can change."""

    inputs_shape = [10, 10]
    inputs = tf.placeholder(tf.float32, shape=[None] + inputs_shape)
    bn = snt.BatchNorm(offset=False, scale=False)

    # Outputs should be equal to inputs.
    out = bn(inputs,
             is_training=False,
             test_local_stats=False)

    init = tf.global_variables_initializer()
    update_ops = tuple(tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    with self.test_session() as sess:
      sess.run(init)

      for batch_size in [1, 3, 10]:
        input_data = np.random.rand(batch_size, *inputs_shape)
        out_v = sess.run(out, feed_dict={inputs: input_data})
        self.assertAllClose(input_data / np.sqrt(1.0 + bn._eps), out_v)

        sess.run(update_ops, feed_dict={inputs: input_data})

  def testInvalidInitializerParameters(self):
    with self.assertRaisesRegexp(KeyError, "Invalid initializer keys.*"):
      snt.BatchNorm(
          initializers={"not_gamma": tf.contrib.layers.l1_regularizer(0.5)})

    err = "Initializer for 'gamma' is not a callable function"
    with self.assertRaisesRegexp(TypeError, err):
      snt.BatchNorm(initializers={"gamma": tf.zeros([1, 2, 3])})

  def testInvalidPartitionerParameters(self):
    with self.assertRaisesRegexp(KeyError, "Invalid partitioner keys.*"):
      snt.BatchNorm(
          partitioners={"not_gamma": tf.contrib.layers.l1_regularizer(0.5)})

    err = "Partitioner for 'gamma' is not a callable function"
    with self.assertRaisesRegexp(TypeError, err):
      snt.BatchNorm(partitioners={"gamma": tf.zeros([1, 2, 3])})

  def testInvalidRegularizationParameters(self):
    with self.assertRaisesRegexp(KeyError, "Invalid regularizer keys.*"):
      snt.BatchNorm(
          regularizers={"not_gamma": tf.contrib.layers.l1_regularizer(0.5)})

    err = "Regularizer for 'gamma' is not a callable function"
    with self.assertRaisesRegexp(TypeError, err):
      snt.BatchNorm(regularizers={"gamma": tf.zeros([1, 2, 3])})

  @parameterized.NamedParameters(
      ("BNNoOffsetScale", False, True),
      ("BNNoOffsetNoScale", False, False),
      ("BNOffsetScale", True, True),
      ("BNOffsetNoScale", True, False),
  )
  def testInitializers(self, offset, scale):
    initializers = {
        "moving_mean": tf.constant_initializer(2.0),
        "moving_variance": tf.constant_initializer(3.0),
    }

    if scale:
      initializers["gamma"] = tf.constant_initializer(4.0)
    if offset:
      initializers["beta"] = tf.constant_initializer(5.0)

    inputs_shape = [10, 10]
    inputs = tf.placeholder(tf.float32, shape=[None] + inputs_shape)
    bn = snt.BatchNorm(offset=offset, scale=scale, initializers=initializers)
    self.assertEqual(bn.initializers, initializers)
    bn(inputs, is_training=True)

    init = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init)

      ones_v = np.ones([1, 1, inputs_shape[-1]])
      self.assertAllClose(bn.moving_mean.eval(), ones_v * 2.0)
      self.assertAllClose(bn.moving_variance.eval(), ones_v * 3.0)

      if scale:
        self.assertAllClose(bn.gamma.eval(), ones_v * 4.0)
      if offset:
        self.assertAllClose(bn.beta.eval(), ones_v * 5.0)

  @parameterized.NamedParameters(
      ("BNNoOffsetScale", False, True),
      ("BNNoOffsetNoScale", False, False),
      ("BNOffsetScale", True, True),
      ("BNOffsetNoScale", True, False),
  )
  def testRegularizersInRegularizationLosses(self, offset, scale):
    regularizers = {}
    if offset:
      regularizers["beta"] = tf.contrib.layers.l1_regularizer(scale=0.5)
    if scale:
      regularizers["gamma"] = tf.contrib.layers.l2_regularizer(scale=0.5)

    inputs_shape = [10, 10]
    inputs = tf.placeholder(tf.float32, shape=[None] + inputs_shape)
    bn = snt.BatchNorm(offset=offset, scale=scale, regularizers=regularizers)
    self.assertEqual(bn.regularizers, regularizers)
    bn(inputs, is_training=True)

    graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    if not offset and not scale:
      self.assertFalse(graph_regularizers)
    if offset and not scale:
      self.assertRegexpMatches(graph_regularizers[0].name, ".*l1_regularizer.*")
    if scale and not offset:
      self.assertRegexpMatches(graph_regularizers[0].name, ".*l2_regularizer.*")
    if scale and offset:
      self.assertRegexpMatches(graph_regularizers[0].name, ".*l1_regularizer.*")
      self.assertRegexpMatches(graph_regularizers[1].name, ".*l2_regularizer.*")

  @parameterized.NamedParameters(
      ("BNNoOffsetScale", False, True),
      ("BNNoOffsetNoScale", False, False),
      ("BNOffsetScale", True, True),
      ("BNOffsetNoScale", True, False),
  )
  def testPartitioners(self, offset, scale):
    partitioners = {}

    if scale:
      partitioners["gamma"] = tf.fixed_size_partitioner(num_shards=2)
    if offset:
      partitioners["beta"] = tf.fixed_size_partitioner(num_shards=2)

    inputs_shape = [10, 10]
    inputs = tf.placeholder(tf.float32, shape=[None] + inputs_shape)
    bn = snt.BatchNorm(offset=offset, scale=scale, partitioners=partitioners)
    self.assertEqual(bn.partitioners, partitioners)
    bn(inputs, is_training=True)

    if scale:
      self.assertEqual(type(bn.gamma), variables.PartitionedVariable)
    if offset:
      self.assertEqual(type(bn.beta), variables.PartitionedVariable)

  @parameterized.NamedParameters(
      ("IsTrainingBoolVal", True, False, False, True),
      ("IsTestingBoolVal", False, True, False, True),
      ("IsTestingBoolValMovingAverage", False, False, False, True),
      ("IsTrainingScaleBoolVal", True, False, True, True),
      ("IsTestingScaleBoolVal", True, True, True, True),
      ("IsTestingScaleBoolValMovingAverage", True, False, True, True),
      ("IsTrainingTensorVal", True, False, False, False),
      ("IsTestingTensorVal", False, True, False, False),
      ("IsTestingTensorValMovingAverage", False, False, False, False),
      ("IsTrainingScaleTensorVal", True, False, True, False),
      ("IsTestingScaleTensorVal", True, True, True, False),
      ("IsTestingScaleTensorValMovingAverage", True, False, True, False))
  def testFusedBatchNorm(self, is_training, test_local_stats, scale,
                         is_training_python_bool):
    input_shape = (32, 9, 9, 8)
    iterations = 5
    x = tf.placeholder(tf.float32, shape=input_shape)
    bn1 = snt.BatchNorm(scale=scale, update_ops_collection=None)

    with self.assertRaises(NotImplementedError):
      # Input does not have 4 dimensions but fused is True.
      xlinear = tf.placeholder(tf.float32, shape=(2, 3))
      snt.BatchNorm(fused=True, scale=scale)(xlinear, is_training=True)

    with self.assertRaises(ValueError):
      # The axis is incorrect
      snt.BatchNorm(axis=(1, 2, 3), fused=True, scale=scale)(
          x, is_training=True)

    bn2 = snt.BatchNorm(scale=scale, fused=True, update_ops_collection=None)

    xx = np.random.random(input_shape)
    feed_dict = {x: xx}
    if not is_training_python_bool:
      is_training_node = tf.placeholder(tf.bool, shape=())
      feed_dict.update({is_training_node: is_training})
      is_training = is_training_node
      test_local_stats_node = tf.placeholder(tf.bool, shape=())
      feed_dict.update({test_local_stats_node: test_local_stats})
      test_local_stats = test_local_stats_node

    o1 = bn1(x, is_training=is_training, test_local_stats=test_local_stats)
    o2 = bn2(x, is_training=is_training, test_local_stats=test_local_stats)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      params = [
          o1, o2, bn1._moving_mean, bn1._moving_variance, bn2._moving_mean,
          bn2._moving_variance
      ]
      for _ in range(iterations):
        y1, y2, mean1, var1, mean2, var2 = sess.run(params, feed_dict=feed_dict)
        self.assertAllClose(y1, y2, atol=1e-4)
        self.assertAllClose(mean1, mean2, atol=1e-4)
        self.assertAllClose(var1, var2, atol=1e-4)


if __name__ == "__main__":
  tf.test.main()
