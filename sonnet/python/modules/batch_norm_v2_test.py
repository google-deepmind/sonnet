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

"""Tests for sonnet.python.modules.batch_norm_v2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Dependency imports
from absl.testing import parameterized
import numpy as np
import sonnet as snt
import tensorflow as tf


class BatchNormV2Test(parameterized.TestCase, tf.test.TestCase):

  def testConstruct(self):
    inputs = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])

    batch_norm1 = snt.BatchNormV2(offset=False, scale=False, fused=False)
    batch_norm1(inputs, is_training=True)

    err = "Batch normalization doesn't have an offset, so no beta"
    with self.assertRaisesRegexp(snt.Error, err):
      _ = batch_norm1.beta

    err = "Batch normalization doesn't have a scale, so no gamma"
    with self.assertRaisesRegexp(snt.Error, err):
      _ = batch_norm1.gamma

    batch_norm2 = snt.BatchNormV2(offset=True, scale=False)
    batch_norm2(inputs, is_training=True)
    _ = batch_norm2.beta

    batch_norm3 = snt.BatchNormV2(offset=False, scale=True)
    batch_norm3(inputs, is_training=True)
    _ = batch_norm3.gamma

    batch_norm4 = snt.BatchNormV2(offset=True, scale=True)
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

  @parameterized.parameters(
      ["NC", "NWC", "NHWC", "NDHWC", "NCW", "NCHW", "NCDHW"])
  def testDataFormats(self, data_format):
    """Check that differing data formats give the correct output shape."""
    dim_sizes = {
        "N": None,
        "D": 10,
        "H": 64,
        "W": 32,
        "C": 3
    }
    inputs = tf.placeholder_with_default(
        tf.zeros([dim_sizes[dim_name] or 5 for dim_name in data_format]),
        [dim_sizes[dim_name] for dim_name in data_format])

    bn_data_formats = [data_format]
    if data_format.endswith("C"):
      bn_data_formats.append(None)

    for bn_data_format in bn_data_formats:
      bn = snt.BatchNormV2(data_format=bn_data_format, offset=False)
      bn(inputs, is_training=True)
      mean_shape = bn.moving_mean.get_shape()
      correct_mean_shape = [
          dim_sizes["C"] if dim_name == "C" else 1 for dim_name in data_format
      ]
      self.assertEqual(mean_shape, correct_mean_shape)

    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu) as sess:
        for bn_data_format in "NC NWC NHWC NDHWC NCW NCHW NCDHW".split():
          if len(data_format) != len(bn_data_format):
            bn = snt.BatchNormV2(data_format=bn_data_format, offset=False)
            err = r"Incorrect data format {} for input shape .*".format(
                bn_data_format)
            with self.assertRaisesRegexp(snt.IncompatibleShapeError, err):
              outputs = bn(inputs, is_training=True)
              sess.run(outputs)

  @parameterized.named_parameters(
      ("Float32", tf.float32),
  )
  def testDataType(self, dtype):
    inputs = tf.placeholder(dtype, shape=[None, 64, 32, 3])
    batch_norm = snt.BatchNormV2(offset=True, scale=True)
    output = batch_norm(inputs, is_training=True)

    self.assertEqual(dtype, output.dtype)
    self.assertEqual(dtype, batch_norm.moving_mean.dtype.base_dtype)
    self.assertEqual(dtype, batch_norm.moving_variance.dtype.base_dtype)
    self.assertEqual(dtype, batch_norm.gamma.dtype.base_dtype)
    self.assertEqual(dtype, batch_norm.beta.dtype.base_dtype)

  @parameterized.named_parameters(
      ("Float16", tf.float16),
      ("BFloat16", tf.bfloat16),
  )
  def test16Bit(self, dtype):
    inputs = tf.placeholder(dtype, shape=[None, 64, 32, 3])
    batch_norm = snt.BatchNormV2(offset=True, scale=True, fused=False)
    output = batch_norm(inputs, is_training=True)

    self.assertEqual(dtype, output.dtype)
    self.assertEqual(tf.float32, batch_norm.moving_mean.dtype.base_dtype)
    self.assertEqual(tf.float32, batch_norm.moving_variance.dtype.base_dtype)
    self.assertEqual(dtype, batch_norm.gamma.dtype.base_dtype)
    self.assertEqual(dtype, batch_norm.beta.dtype.base_dtype)

  def _get_inputs(self, dtype=tf.float32):
    v = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=dtype.as_numpy_dtype)
    input_v = np.array([v, v, v, v, v, v, v])
    inputs = tf.constant(input_v)

    return v, input_v, inputs

  def testUpdateImproveStatistics(self):
    """Test that updating the moving_mean improves statistics."""

    _, _, inputs = self._get_inputs()

    # Use small decay_rate to update faster.
    bn = snt.BatchNormV2(
        offset=False,
        scale=False,
        decay_rate=0.1,
        update_ops_collection=tf.GraphKeys.UPDATE_OPS)
    out1 = bn(inputs, is_training=False, test_local_stats=False)

    # Build the update ops.
    bn(inputs, is_training=True)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      out_v = sess.run(out1)

      # Before updating the moving_mean the results are off.
      self.assertBetween(np.max(np.abs(np.zeros([7, 6]) - out_v)), 2, 5)

      sess.run(tuple(tf.get_collection(tf.GraphKeys.UPDATE_OPS)))

      # After updating the moving_mean the results are better.
      out_v = sess.run(out1)
      self.assertBetween(np.max(np.abs(np.zeros([7, 6]) - out_v)), 1, 2)

  @parameterized.named_parameters(
      ("Float16", tf.float16),
      ("Float32", tf.float32),
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
    bn = snt.BatchNormV2(
        offset=False,
        scale=False,
        decay_rate=0.9,
        update_ops_collection=tf.GraphKeys.UPDATE_OPS)

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

      atol = 1.e-2 if dtype == tf.float16 else 1.e-6

      self.assertAllClose(np.reshape(correct_mm, [1, 6]), mm, atol=atol)
      self.assertAllClose(np.reshape(correct_mv, [1, 6]), mv, atol=atol)

  def testCheckStatsPython(self):
    """The correct normalization is being used for different Python flags."""

    v, input_v, inputs = self._get_inputs()

    bn = snt.BatchNormV2(
        offset=False,
        scale=False,
        decay_rate=0.5,
        update_ops_collection=tf.GraphKeys.UPDATE_OPS
    )
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
      self.assertAllClose(np.zeros([7, 6]), out2_, rtol=1e-6, atol=1e-5)

      # Out3: Tested using moving average stats.
      self.assertAllClose(
          (input_v - mm) / np.sqrt(mv + bn._eps),
          out3_)

  @parameterized.named_parameters(
      ("UseUpdateCollection", tf.GraphKeys.UPDATE_OPS),
      ("UseDifferentUpdateCollection", "my_update_ops"),
      ("UseControlDependencies", None),
  )
  def testCheckStatsInGraph(self, update_ops_collection):
    """The correct normalization is being used for different TF flags."""

    v, input_v, inputs = self._get_inputs()

    bn = snt.BatchNormV2(
        offset=False,
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

      self.assertAllClose(np.zeros([7, 6]), out_v, rtol=1e-6, atol=1e-5)

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

    bn = snt.BatchNormV2(
        offset=True,
        scale=True,
        update_ops_collection=tf.GraphKeys.UPDATE_OPS)

    bn(inputs1, is_training=True)
    bn(inputs2, is_training=False)

    self.assertEqual(len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)), 4)

    # We should have one set of update ops
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    self.assertEqual(len(update_ops), 2)

  def testUpdatesInsideCond(self):
    """Demonstrate that updates inside a cond fail."""

    _, input_v, inputs = self._get_inputs()
    bn = snt.BatchNormV2(
        offset=False,
        scale=False,
        decay_rate=0.5,
        update_ops_collection=tf.GraphKeys.UPDATE_OPS)
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
    bn = snt.BatchNormV2(
        offset=False, scale=False)

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
      snt.BatchNormV2(
          initializers={"not_gamma": tf.contrib.layers.l1_regularizer(0.5)})

    err = "Initializer for 'gamma' is not a callable function"
    with self.assertRaisesRegexp(TypeError, err):
      snt.BatchNormV2(initializers={"gamma": tf.zeros([1, 2, 3])})

  def testInvalidPartitionerParameters(self):
    with self.assertRaisesRegexp(KeyError, "Invalid partitioner keys.*"):
      snt.BatchNormV2(
          partitioners={"not_gamma": tf.contrib.layers.l1_regularizer(0.5)})

    err = "Partitioner for 'gamma' is not a callable function"
    with self.assertRaisesRegexp(TypeError, err):
      snt.BatchNormV2(partitioners={"gamma": tf.zeros([1, 2, 3])})

  def testInvalidRegularizationParameters(self):
    with self.assertRaisesRegexp(KeyError, "Invalid regularizer keys.*"):
      snt.BatchNormV2(
          regularizers={"not_gamma": tf.contrib.layers.l1_regularizer(0.5)})

    err = "Regularizer for 'gamma' is not a callable function"
    with self.assertRaisesRegexp(TypeError, err):
      snt.BatchNormV2(regularizers={"gamma": tf.zeros([1, 2, 3])})

  @parameterized.named_parameters(
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
    bn = snt.BatchNormV2(
        offset=offset,
        scale=scale,
        initializers=initializers)
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

  @parameterized.named_parameters(
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
    bn = snt.BatchNormV2(
        offset=offset,
        scale=scale,
        regularizers=regularizers)
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

  @parameterized.named_parameters(
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
    bn = snt.BatchNormV2(
        offset=offset,
        scale=scale,
        partitioners=partitioners)
    self.assertEqual(bn.partitioners, partitioners)
    bn(inputs, is_training=True)

    if scale:
      self.assertLen(tf.global_variables("batch_norm/gamma"), 2)
    if offset:
      self.assertLen(tf.global_variables("batch_norm/beta"), 2)

  @parameterized.named_parameters(
      ("IsTrainingBoolVal", True, False, False, True),
      ("IsTestingBoolVal", False, True, False, True),
      ("IsTestingBoolValMovingAverage", False, False, False, True),
      ("IsTrainingScaleBoolVal", True, False, True, True),
      ("IsTestingScaleBoolVal", False, True, True, True),
      ("IsTestingScaleBoolValMovingAverage", False, False, True, True),
      ("IsTrainingTensorVal", True, False, False, False),
      ("IsTestingTensorVal", False, True, False, False),
      ("IsTestingTensorValMovingAverage", False, False, False, False),
      ("IsTrainingScaleTensorVal", True, False, True, False),
      ("IsTestingScaleTensorVal", False, True, True, False),
      ("IsTestingScaleTensorValMovingAverage", False, False, True, False))
  def testFusedBatchNormV2(self, is_training, test_local_stats, scale,
                           is_training_python_bool):
    input_shape = (32, 9, 9, 8)
    iterations = 5
    x = tf.placeholder(tf.float32, shape=input_shape)
    bn1 = snt.BatchNormV2(scale=scale)
    bn2 = snt.BatchNormV2(fused=False, scale=scale)

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

  @parameterized.named_parameters(
      ("IsTraining", True, False),
      ("IsTesting", False, True),
      ("IsTestingMovingAverage", False, False))
  def testFusedBatchNormFloat16(self, is_training, test_local_stats):
    input_shape = (31, 7, 7, 5)
    iterations = 3
    x = tf.placeholder(tf.float16, shape=input_shape)
    bn1 = snt.BatchNormV2(fused=False)
    bn2 = snt.BatchNormV2()

    feed_dict = {x: np.random.random(input_shape)}

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
        self.assertAllClose(y1, y2, atol=1e-2)
        self.assertAllClose(mean1, mean2, atol=1e-2)
        self.assertAllClose(var1, var2, atol=1e-2)

  def testCheckpointCompatibility(self):
    save_path = os.path.join(self.get_temp_dir(), "basic_save_restore")

    input_shape_1 = (31, 7, 7, 5)
    input_shape_2 = (31, 5, 7, 7)

    x1 = tf.placeholder(tf.float32, shape=input_shape_1)
    bn1 = snt.BatchNormV2(data_format="NHWC")
    bn1(x1, is_training=True)
    saver1 = snt.get_saver(bn1)

    x2 = tf.placeholder(tf.float32, shape=input_shape_2)
    bn2 = snt.BatchNormV2(data_format="NCHW")
    bn2(x2, is_training=False)
    saver2 = snt.get_saver(bn2)

    x3 = tf.placeholder(tf.float32, shape=input_shape_1)
    bn3 = snt.BatchNormV2(data_format="NCHW")
    bn3(x3, is_training=False)
    saver3 = snt.get_saver(bn3)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      saver1.save(sess, save_path)
      saver2.restore(sess, save_path)
      with self.assertRaises(tf.errors.InvalidArgumentError):
        saver3.restore(sess, save_path)


if __name__ == "__main__":
  tf.test.main()
