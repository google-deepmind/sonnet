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

"""Batch normalization module for Sonnet.

This contains the module BatchNormV2, which performs batch normalization on
its inputs. It has an optional post-normalization scale and offset, and it
maintains moving averages of the statistics for use at test time.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports


from sonnet.python.modules import base
from sonnet.python.modules import conv
from sonnet.python.modules import util

import tensorflow as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.layers import utils
from tensorflow.python.training import moving_averages
# pylint: enable=g-direct-tensorflow-import


def create_beta_initializer():
  """Returns a default initializer for the `beta` in batch norm."""
  return tf.zeros_initializer()


def create_gamma_initializer():
  """Returns a default initializer for the `gamma` in batch norm."""
  return tf.ones_initializer()


def create_mean_initializer():
  """Returns a default initializer for the `moving_mean` in batch norm."""
  return tf.zeros_initializer()


def create_variance_initializer():
  """Returns a default initializer for the `moving_variance` in batch norm."""
  return tf.ones_initializer()


class BatchNormV2(base.AbstractModule):
  """Batch normalization module, including optional affine transformation.

  This module maintains exponential moving averages of the mean and
  variance, which can be optionally used to normalize at test time.

  At training time, batch statistics (mean, variance) are not shared between
  separate connections. The moving averages are shared between separate
  connections. At both training and test time, the optional affine
  transformation (`* gamma + beta`) is shared between separate connections.

  This is also the case for distributed replica training, where the batch
  statistics are not aggregated across replicas, but the moving averages are
  shared globally.

  When connecting the module to the graph, `is_training=True` means that

    - Update ops are created to update the moving averages with the current
      batch's statistics.
    - Features are normalized using the *current batch's statistics*. The
      `test_local_stats` setting is ignored. The moving averages are
      **not** used.

  whereas `is_training=False` means that

    - Update ops are not created.
    - Features are normalized using either:
      - The moving averages if `test_local_stats=False` (default).
      - The test batch statistics if `test_local_stats=True`.

  The moving averages are used by default at test time, but local batch
  statistics can be used by specifying a flag when connecting. One often wants
  to use local batch statistics at test time to track the progress while the
  model is trained as it would ensure that moving average updates do not affect
  the training curves. Once the training is finished, it's often advantageous
  to use moving average statistics, since it would make evaluation agnostic to
  the batch size, and might even lead to small improvements over the local
  batch statistics.

  The moving averages will be updated automatically by default, but not if
  `update_ops_collection` is provided: in that case they will only be updated
  when the ops in that collection are run.

  For example, to run the updates automatically:

      bn = BatchNormV2()
      train_net = bn(train_inputs, is_training=True)

  this does, however, have the effect of blocking the forwards pass of the
  network until the update ops have been run and may have a small performance
  penalty.

  For example, to run the updates manually:

      bn = BatchNormV2(update_ops_collection=tf.GraphKeys.UPDATE_OPS)
      train_net = bn(train_inputs, is_training=True)

      ...

      update_ops = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
      train_op = tf.group(train_op, update_ops)

  Then, whenever `train_op` is run so also are the moving average update ops.

  Some batch normalization caveats:

    - Batch normalization will remove the effect of adding a bias, so e.g.
      `use_bias=False` should be used for an immediately preceding snt.Linear
      module.
    - If your data batches aren't i.i.d. then batch normalization can allow your
      network to 'cheat' by using the batch statistics to peek at the rest of
      the batch. This can exhibit itself as a higher test score with
      `test_local_stats=True` than `test_local_stats=False`.
  """

  GAMMA = "gamma"
  BETA = "beta"
  MOVING_MEAN = "moving_mean"
  MOVING_VARIANCE = "moving_variance"
  POSSIBLE_INITIALIZER_KEYS = {GAMMA, BETA, MOVING_MEAN, MOVING_VARIANCE}
  POSSIBLE_PARTITIONER_KEYS = {GAMMA, BETA}
  POSSIBLE_REGULARIZER_KEYS = {GAMMA, BETA}

  SUPPORTED_DATA_FORMATS = set.union({"NC"},
                                     conv.SUPPORTED_1D_DATA_FORMATS,
                                     conv.SUPPORTED_2D_DATA_FORMATS,
                                     conv.SUPPORTED_3D_DATA_FORMATS)

  def __init__(self, data_format=None, offset=True, scale=False,
               decay_rate=0.999, eps=1e-3, initializers=None,
               partitioners=None, regularizers=None,
               update_ops_collection=None, fused=True,
               name="batch_norm"):
    """Constructs a BatchNormV2 module.

    Reduces over all input tensor dimensions apart from the channel
    dimension. This has the effect of treating pixels in 1D/2D/3D images as
    additional elements of the minibatch.

    Args:
      data_format: The data format. Can be "NC", "NWC", "NCW", "NHWC", "NCHW",
        "NDHWC", or "NCDHW". If not provided we assume the channel dimension is
        last.
      offset: Optional boolean to specify whether or not to apply a trained
        component-wise bias after the batch normalization and scaling.
      scale: Optional boolean to specify whether or not to apply a trained
        component-wise scale after the batch normalization.
      decay_rate: Decay rate of the exponential moving averages of the mean
        and variance.
      eps: Small number to avoid dividing by zero when diving by the standard
        deviation.
      initializers: Optional dict containing ops to initialize the weights of
        the affine transform (`gamma` and `beta`).
      partitioners: Optional dict containing partitioners to partition the
        weights of the affine transform (`gamma` and `beta`).
      regularizers: Optional dict containing regularizers for the weights of the
        affine transform ("gamma" and "beta"). As a default, no regularizers are
        used. A regularizer should be a function that takes a single `Tensor` as
        an input and returns a scalar `Tensor` output, e.g. the L1 and L2
        regularizers in `tf.contrib.layers`.
      update_ops_collection: Optional name of TensorFlow variable collection to
        add the moving average update ops to. If not provided, we instead add
        the update ops as control dependencies of the output of the module. This
        may result in some slowdown, as the feed-forward of the network is now
        blocked.
      fused: Use nn.fused_batch_norm if True, nn.batch_normalization otherwise.
      name: Name of the module.

    Raises:
      KeyError: If `initializers` contains any keys other than `gamma`, `beta`,
        `moving_mean` or `moving_variance`.
      KeyError: If `partitioners` or `regularizers` contains any keys other
        than `gamma` or `beta`.
      TypeError: If any of the given initializers, partitioners or regularizers
        are not callable.
      ValueError: If `data_format` is invalid.
    """
    super(BatchNormV2, self).__init__(name=name)

    if data_format not in self.SUPPORTED_DATA_FORMATS.union({None}):
      raise ValueError("Invalid data_format: %r" % (data_format,))

    self._data_format = data_format
    self._offset = offset
    self._scale = scale
    self._decay_rate = decay_rate
    self._eps = eps
    self._update_ops_collection = update_ops_collection
    self._fused = fused

    self._initializers = util.check_initializers(
        initializers, self.POSSIBLE_INITIALIZER_KEYS)
    self._partitioners = util.check_partitioners(
        partitioners, self.POSSIBLE_PARTITIONER_KEYS)
    self._regularizers = util.check_regularizers(
        regularizers, self.POSSIBLE_REGULARIZER_KEYS)

  def _build_statistics(self, input_batch, use_batch_stats, stat_dtype):
    """Builds the statistics part of the graph when using moving variance.

    Args:
      input_batch: Input batch Tensor.
      use_batch_stats: Boolean to indicate if batch statistics should be
        calculated, otherwise moving averages are returned.
      stat_dtype: TensorFlow datatype to use for the moving mean and variance.

    Returns:
      Tuple of (mean, variance), each of the same datatype as `input_batch`.
    """
    # Set up our moving statistics. When connecting in parallel, this is shared.
    if self.MOVING_MEAN not in self._initializers:
      self._initializers[self.MOVING_MEAN] = create_mean_initializer()
    self._moving_mean = tf.get_variable(
        "moving_mean",
        dtype=stat_dtype,
        shape=(self._num_channels,),
        collections=[
            tf.GraphKeys.MOVING_AVERAGE_VARIABLES,
            tf.GraphKeys.GLOBAL_VARIABLES,
        ],
        initializer=self._initializers[self.MOVING_MEAN],
        trainable=False)

    if self.MOVING_VARIANCE not in self._initializers:
      self._initializers[self.MOVING_VARIANCE] = create_variance_initializer()
    self._moving_variance = tf.get_variable(
        "moving_variance",
        dtype=stat_dtype,
        shape=(self._num_channels,),
        collections=[
            tf.GraphKeys.MOVING_AVERAGE_VARIABLES,
            tf.GraphKeys.GLOBAL_VARIABLES,
        ],
        initializer=self._initializers[self.MOVING_VARIANCE],
        trainable=False)

    def build_batch_stats():
      """Builds the batch statistics calculation ops."""
      mean, variance = tf.nn.moments(input_batch, self._axis,
                                     keep_dims=True, name="normalize_moments")

      return mean, variance

    def build_moving_stats():
      """Retrieves the moving statistics."""
      # If necessary, cast the moving statistics to match the input type.
      # This is required by tf.nn.batch_normalization.
      input_dtype = input_batch.dtype.base_dtype
      if stat_dtype == input_dtype:
        return (
            tf.identity(self._moving_mean),
            tf.identity(self._moving_variance),
        )
      else:
        return (
            tf.cast(self._moving_mean, input_dtype),
            tf.cast(self._moving_variance, input_dtype),
        )

    mean, variance = tf.contrib.framework.smart_cond(
        use_batch_stats,
        build_batch_stats,
        build_moving_stats,
    )

    return mean, variance

  def _build_update_ops(self, mean, variance, is_training):
    """Builds the moving average update ops when using moving variance.

    Args:
      mean: The mean value to update with.
      variance: The variance value to update with.
      is_training: Boolean Tensor to indicate if we're currently in
        training mode.

    Returns:
      Tuple of `(update_mean_op, update_variance_op)` when `is_training` is or
      could be `True`. Returns `None` when `is_training=False`.
    """

    def build_update_ops():
      """Builds the exponential moving average update ops."""

      update_mean_op = moving_averages.assign_moving_average(
          variable=self._moving_mean,
          value=tf.reshape(mean, (self._num_channels,)),
          decay=self._decay_rate,
          zero_debias=False,
          name="update_moving_mean").op

      update_variance_op = moving_averages.assign_moving_average(
          variable=self._moving_variance,
          value=tf.reshape(variance, (self._num_channels,)),
          decay=self._decay_rate,
          zero_debias=False,
          name="update_moving_variance").op

      return update_mean_op, update_variance_op

    def build_no_ops():
      return (tf.no_op(), tf.no_op())

    # Only make the ops if we know that `is_training=True`, or the value of
    # `is_training` is unknown.
    is_training_const = utils.constant_value(is_training)
    if is_training_const is None or is_training_const:
      update_mean_op, update_variance_op = tf.contrib.framework.smart_cond(
          is_training,
          build_update_ops,
          build_no_ops,
      )
      return (update_mean_op, update_variance_op)
    else:
      return None

  def _fused_batch_norm_op(self, input_batch, mean, variance, use_batch_stats):
    """Creates a fused batch normalization op."""
    # Store the original shape of the mean and variance.
    mean_shape = mean.get_shape()
    variance_shape = variance.get_shape()
    # The fused batch norm expects the mean, variance, gamma and beta
    # tensors to have dimension 1, so we flatten them to remove the
    # extra dimensions. In addition, it expects the input_batch to have
    # dimension 4, so we reshape it accordingly.
    gamma_flatten = tf.reshape(self._gamma, shape=(self._num_channels,))
    beta_flatten = tf.reshape(self._beta, shape=(self._num_channels,))
    flatten_mean = tf.reshape(mean, shape=(self._num_channels,))
    flatten_variance = tf.reshape(variance, shape=(self._num_channels,))
    use_batch_stats = tf.convert_to_tensor(use_batch_stats)

    input_shape = input_batch.get_shape()
    output_shape = tf.shape(input_batch)

    flat_image_size = tf.cast(tf.reduce_prod(self._image_shape, keepdims=True),
                              tf.int64)

    if len(self._data_format) == 4:
      fusable_data_format = self._data_format
      fusable_batch = input_batch
    elif self._channel_index == 1 and input_shape.rank > 2:
      fusable_data_format = "NCHW"
      fusable_shape = tf.concat(
          [[-1, self._num_channels, 1], flat_image_size], axis=0)
      fusable_batch = tf.reshape(input_batch, shape=fusable_shape)
    else:
      # The CPU implementation of FusedBatchNorm only supports NHWC tensor
      # format for now.
      fusable_data_format = "NHWC"
      fusable_shape = tf.concat(
          [[-1, 1], flat_image_size, [self._num_channels]], axis=0)
      fusable_batch = tf.reshape(input_batch, shape=fusable_shape)

    common_args = {
        "scale": gamma_flatten,
        "offset": beta_flatten,
        "epsilon": self._eps,
        "data_format": fusable_data_format,
        "name": "batch_norm"
    }

    def use_batch_stats_fused_batch_norm():
      return tf.nn.fused_batch_norm(
          fusable_batch,
          mean=None,
          variance=None,
          is_training=True,
          **common_args)

    def moving_average_fused_batch_norm():
      return tf.nn.fused_batch_norm(
          fusable_batch,
          mean=flatten_mean,
          variance=flatten_variance,
          is_training=False,
          **common_args)

    batch_norm_op, mean, variance = tf.contrib.framework.smart_cond(
        use_batch_stats, use_batch_stats_fused_batch_norm,
        moving_average_fused_batch_norm)

    if len(self._data_format) != 4:
      batch_norm_op = tf.reshape(batch_norm_op, output_shape)
    mean = tf.reshape(mean, mean_shape)
    variance = tf.reshape(variance, variance_shape)
    return batch_norm_op, mean, variance

  def _batch_norm_op(self, input_batch, mean, variance, use_batch_stats,
                     stat_dtype):
    """Creates a batch normalization op.

    It uses the tf.nn.batch_normalization op by default and the
    tf.nn.fused_batch_norm op to support fused batch normalization.

    Args:
      input_batch: A input Tensor of arbitrary dimension.
      mean: A mean tensor, of the same dtype as `input_batch`.
      variance: A variance tensor, of the same dtype as `input_batch`.
      use_batch_stats: A bool value that indicates whether the operation should
         use the batch statistics.
      stat_dtype: TensorFlow datatype used for the moving mean and variance.

    Returns:
      A batch normalization operation.
      The current mean tensor, of datatype `stat_dtype`.
      The current variance tensor, of datatype `stat_dtype`.
    """
    if self._fused:
      # For the non-training case where not using batch stats,
      # pass in the moving statistic variables directly.
      # These will already be in the correct dtype, even for float16 input.
      batch_norm_op, mean, variance = self._fused_batch_norm_op(
          input_batch,
          self._moving_mean, self._moving_variance, use_batch_stats)
    else:
      if self._beta is None:
        beta = None
      else:
        beta = tf.reshape(self._beta, self._expanded_mean_shape)

      if self._gamma is None:
        gamma = None
      else:
        gamma = tf.reshape(self._gamma, self._expanded_mean_shape)

      batch_norm_op = tf.nn.batch_normalization(
          input_batch,
          tf.reshape(mean, self._expanded_mean_shape),
          tf.reshape(variance, self._expanded_mean_shape),
          beta,
          gamma,
          self._eps,
          name="batch_norm")
      # We'll echo the supplied mean and variance so that they can also be used
      # to update the moving statistics. Cast to matching type if necessary.
      if input_batch.dtype.base_dtype != stat_dtype:
        mean = tf.cast(mean, stat_dtype)
        variance = tf.cast(variance, stat_dtype)

    return batch_norm_op, mean, variance

  def _build_scale_offset(self, dtype):
    """Sets up optional scale and offset factors."""

    # tf.nn.fused_batch_norm accepts float16 batch data, but not scale/offset.
    if self._fused and dtype == tf.float16:
      dtype = tf.float32

    # The fused batch norm operation needs the beta, gamma variables,
    # so in this case we build them and set the trainable option according
    # to the values of _offset and _scale.
    self._beta = None
    if self._offset or self._fused:
      if self.BETA not in self._initializers:
        self._initializers[self.BETA] = create_beta_initializer()
      self._beta = tf.get_variable(
          self.BETA,
          dtype=dtype,
          shape=(self._num_channels,),
          initializer=self._initializers[self.BETA],
          partitioner=self._partitioners.get(self.BETA, None),
          regularizer=self._regularizers.get(self.BETA, None),
          trainable=self._offset)

    self._gamma = None
    if self._scale or self._fused:
      if self.GAMMA not in self._initializers:
        self._initializers[self.GAMMA] = create_gamma_initializer()
      self._gamma = tf.get_variable(
          self.GAMMA,
          dtype=dtype,
          shape=(self._num_channels,),
          initializer=self._initializers[self.GAMMA],
          partitioner=self._partitioners.get(self.GAMMA, None),
          regularizer=self._regularizers.get(self.GAMMA, None),
          trainable=self._scale)

  def _build(self,
             input_batch,
             is_training,
             test_local_stats=False):
    """Connects the BatchNormV2 module into the graph.

    Args:
      input_batch: A Tensor of the same dimension as `len(data_format)`.
      is_training: A boolean to indicate if the module should be connected in
        training mode, meaning the moving averages are updated. Can be a Tensor.
      test_local_stats: A boolean to indicate if local batch statistics should
        be used when `is_training=False`. If not, moving averages are used.
        By default `False`. Can be a Tensor.

    Returns:
      A tensor with the same shape as `input_batch`.

    Raises:
      base.IncompatibleShapeError: If `data_format` is not valid for the
        input shape.
      base.NotSupportedError: If `input_batch` has data type of `tf.bfloat16`.
    """
    input_shape = input_batch.get_shape()
    input_shape_list = input_shape.as_list()
    input_shape_len = len(input_shape_list)

    if not self._data_format:
      if input_shape_len == 2:
        self._data_format = "NC"
      elif input_shape_len == 3:
        self._data_format = "NWC"
      elif input_shape_len == 4:
        self._data_format = "NHWC"
      elif input_shape_len == 5:
        self._data_format = "NDHWC"
      else:
        raise base.IncompatibleShapeError(
            "Input shape {} has too many or too few dimensions.".format(
                input_shape))

    self._channel_index = self._data_format.index("C")
    # Use list to turn range into iterator in python3.
    self._axis = list(range(len(self._data_format)))
    del self._axis[self._channel_index]

    if len(self._data_format) != input_shape_len:
      raise base.IncompatibleShapeError(
          "Incorrect data format {} for input shape {}.".format(
              self._data_format, input_shape))

    dtype = input_batch.dtype.base_dtype
    if self._fused and dtype == tf.bfloat16:
      raise base.NotSupportedError(
          "Fused batch norm does not support tf.bfloat16.")
    # Maintain moving averages at a minimum precision of tf.float32.
    stat_dtype = tf.float32 if dtype in [tf.float16, tf.bfloat16] else dtype

    self._num_channels = int(input_shape_list[self._channel_index])
    if self._channel_index == 1:
      spatial_dimensions_slice = slice(2, input_shape_len)
    else:
      assert self._channel_index == (input_shape_len - 1)
      spatial_dimensions_slice = slice(1, -1)
    # If the spatial dimensions are not fully defined, then self._image_shape
    # has to be a Tensor instead of a list of python ints.
    if input_shape[spatial_dimensions_slice].is_fully_defined():
      self._image_shape = [
          int(x) for x in input_shape[spatial_dimensions_slice]]
    else:
      self._image_shape = tf.shape(input_batch)[spatial_dimensions_slice]
    self._expanded_mean_shape = [1] * input_shape_len
    self._expanded_mean_shape[self._channel_index] = self._num_channels

    use_batch_stats = is_training | test_local_stats

    mean, variance = self._build_statistics(input_batch, use_batch_stats,
                                            stat_dtype)

    # Sets up optional gamma and beta parameters
    self._build_scale_offset(dtype)
    # Sets up the batch normalization op.
    out, mean, variance = self._batch_norm_op(input_batch, mean, variance,
                                              use_batch_stats, stat_dtype)
    # Sets up the update op.
    update_ops = self._build_update_ops(mean, variance, is_training)

    # Put update ops in the update ops collection if given, otherwise add as
    # control dependencies of the output.
    if update_ops:
      if self._update_ops_collection:
        for update_op in update_ops:
          tf.add_to_collection(self._update_ops_collection, update_op)
      else:
        with tf.control_dependencies(update_ops):
          out = tf.identity(out)

    return out

  @property
  def initializers(self):
    return self._initializers

  @property
  def partitioners(self):
    return self._partitioners

  @property
  def regularizers(self):
    return self._regularizers

  @property
  def moving_mean(self):
    self._ensure_is_connected()
    return tf.reshape(self._moving_mean, self._expanded_mean_shape)

  @property
  def moving_variance(self):
    self._ensure_is_connected()
    return tf.reshape(self._moving_variance, self._expanded_mean_shape)

  @property
  def beta(self):
    self._ensure_is_connected()

    if self._beta is None:
      raise base.Error(
          "Batch normalization doesn't have an offset, so no beta")
    else:
      return tf.reshape(self._beta, self._expanded_mean_shape)

  @property
  def gamma(self):
    self._ensure_is_connected()

    if self._gamma is None:
      raise base.Error(
          "Batch normalization doesn't have a scale, so no gamma")
    else:
      return tf.reshape(self._gamma, self._expanded_mean_shape)
