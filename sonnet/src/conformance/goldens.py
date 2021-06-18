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
"""Golden test cases."""

import abc
from typing import Sequence, Tuple

from absl.testing import parameterized
import numpy as np
import sonnet as snt
import tensorflow as tf

_all_goldens = []


def named_goldens() -> Sequence[Tuple[str, "Golden"]]:
  return ((name, cls()) for _, name, cls in list_goldens())


def all_goldens(test_method):
  return parameterized.named_parameters(named_goldens())(test_method)


def _register_golden(module_cls, golden_name):

  def registration_fn(golden_cls):
    _all_goldens.append((module_cls, golden_name, golden_cls))
    golden_cls.name = golden_name
    return golden_cls

  return registration_fn


def list_goldens():
  return list(_all_goldens)


def range_like(t, start=0):
  """Returns a tensor with sequential values of the same dtype/shape as `t`.

  >>> range_like(tf.ones([2, 2]))
  <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
  array([[ 0.,  1.],
         [ 2.,  3.]], dtype=float32)>

  >>> range_like(tf.ones([2, 2]), start=5)
  <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
  array([[ 5.,  6.],
         [ 7.,  8.]], dtype=float32)>

  Args:
    t: A tensor like object (with shape and dtype).
    start: Value to start the range from.

  Returns:
    A `tf.Tensor` with sequential element values the same shape/dtype as `t`.
  """
  return tf.reshape(
      tf.cast(
          tf.range(start,
                   np.prod(t.shape, dtype=int) + start), dtype=t.dtype),
      t.shape)


class Golden(abc.ABC):
  """Represents a golden checkpoint file."""

  @abc.abstractmethod
  def create_module(self):
    """Should create a new module instance and return it."""
    pass

  @abc.abstractmethod
  def create_all_variables(self, module):
    """Create all variables for the given model and return them."""
    pass

  @abc.abstractmethod
  def forward(self, module, x=None):
    """Return the output from calling the module with a fixed input."""
    pass


class AbstractGolden(Golden):
  """Abstract base class for golden tests of single input modules."""

  deterministic = True

  has_side_effects = False

  # Tolerance to be used for assertAllClose calls on TPU, where lower precision
  # can mean results differ more.
  tpu_atol = 1e-3

  @abc.abstractproperty
  def input_spec(self):
    pass

  @abc.abstractproperty
  def num_variables(self):
    pass

  def forward(self, module, x=None):
    if x is None:
      x = range_like(self.input_spec, start=1)
    return module(x)

  def create_all_variables(self, module):
    self.forward(module)
    variables = module.variables
    assert len(variables) == self.num_variables, (
        "Expected %d params, got %d %r" %
        (self.num_variables, len(variables), variables))
    return variables


# pylint: disable=missing-docstring
@_register_golden(snt.Linear, "linear_1x1")
class Linear1x1Test(AbstractGolden):
  create_module = lambda _: snt.Linear(1)
  input_spec = tf.TensorSpec([128, 1])
  num_variables = 2


@_register_golden(snt.Linear, "linear_nobias_1x1")
class LinearNoBias1x1(AbstractGolden):
  create_module = lambda _: snt.Linear(1, with_bias=False)
  input_spec = tf.TensorSpec([1, 1])
  num_variables = 1


@_register_golden(snt.Conv1D, "conv1d_3x3_2x2")
class Conv1D(AbstractGolden):
  create_module = lambda _: snt.Conv1D(output_channels=3, kernel_shape=3)
  input_spec = tf.TensorSpec([1, 2, 2])
  num_variables = 2


@_register_golden(snt.Conv2D, "conv2d_3x3_2x2")
class Conv2D(AbstractGolden):
  create_module = lambda _: snt.Conv2D(output_channels=3, kernel_shape=3)
  input_spec = tf.TensorSpec([1, 2, 2, 2])
  num_variables = 2


@_register_golden(snt.Conv3D, "conv3d_3x3_2x2")
class Conv3D(AbstractGolden):
  create_module = lambda _: snt.Conv3D(output_channels=3, kernel_shape=3)
  input_spec = tf.TensorSpec([1, 2, 2, 2, 2])
  num_variables = 2


@_register_golden(snt.Conv1DTranspose, "conv1d_transpose_3x3_2x2")
class Conv1DTranspose(AbstractGolden):
  create_module = (
      lambda _: snt.Conv1DTranspose(output_channels=3, kernel_shape=3))
  input_spec = tf.TensorSpec([1, 2, 2])
  num_variables = 2


@_register_golden(snt.Conv2DTranspose, "conv2d_transpose_3x3_2x2")
class Conv2DTranspose(AbstractGolden):
  create_module = (
      lambda _: snt.Conv2DTranspose(output_channels=3, kernel_shape=3))
  input_spec = tf.TensorSpec([1, 2, 2, 2])
  num_variables = 2


@_register_golden(snt.Conv3DTranspose, "conv3d_transpose_3x3_2x2")
class Conv3DTranspose(AbstractGolden):
  create_module = (
      lambda _: snt.Conv3DTranspose(output_channels=3, kernel_shape=3))
  input_spec = tf.TensorSpec([1, 2, 2, 2, 2])
  num_variables = 2


@_register_golden(snt.DepthwiseConv2D, "depthwise_conv2d_3x3_2x2")
class DepthwiseConv2D(AbstractGolden):
  create_module = lambda _: snt.DepthwiseConv2D(kernel_shape=3)
  input_spec = tf.TensorSpec([1, 2, 2, 2])
  num_variables = 2


@_register_golden(snt.nets.MLP, "mlp_3x4x5_1x3")
class MLP(AbstractGolden):
  create_module = (lambda _: snt.nets.MLP([3, 4, 5]))
  input_spec = tf.TensorSpec([1, 3])
  num_variables = 6


@_register_golden(snt.nets.MLP, "mlp_nobias_3x4x5_1x3")
class MLPNoBias(AbstractGolden):
  create_module = (lambda _: snt.nets.MLP([3, 4, 5], with_bias=False))
  input_spec = tf.TensorSpec([1, 3])
  num_variables = 3


@_register_golden(snt.nets.Cifar10ConvNet, "cifar10_convnet_2x3_2x2_1x3x3x2")
class Cifar10ConvNet(AbstractGolden):
  create_module = (
      lambda _: snt.nets.Cifar10ConvNet(output_channels=(2, 3), strides=(2, 2)))
  input_spec = tf.TensorSpec([1, 3, 3, 2])
  num_variables = 22

  def forward(self, module, x=None):
    if x is None:
      x = range_like(self.input_spec, start=1)
    return module(x, is_training=False, test_local_stats=True)["logits"]


@_register_golden(snt.LayerNorm, "layer_norm_1_1x3_2")
class LayerNorm(AbstractGolden):
  create_module = (
      lambda _: snt.LayerNorm(1, create_scale=True, create_offset=True))
  input_spec = tf.TensorSpec([1, 3, 2])
  num_variables = 2


@_register_golden(snt.InstanceNorm, "instance_norm_1_1x3_2")
class Instance(AbstractGolden):
  create_module = (
      lambda _: snt.InstanceNorm(create_scale=True, create_offset=True))
  input_spec = tf.TensorSpec([1, 3, 2])
  num_variables = 2


@_register_golden(snt.GroupNorm, "group_norm_2_1x3x4")
class GroupNorm(AbstractGolden):
  create_module = (
      lambda _: snt.GroupNorm(2, create_scale=True, create_offset=True))
  input_spec = tf.TensorSpec([1, 3, 4])
  num_variables = 2


@_register_golden(snt.BaseBatchNorm, "base_batch_norm_1x2x2x3")
class BaseBatchNorm(AbstractGolden):
  create_module = (
      lambda _: snt.BaseBatchNorm(True, False, FooMetric(), FooMetric()))  # pytype: disable=wrong-arg-types
  input_spec = tf.TensorSpec([1, 2, 2, 3])
  num_variables = 2

  def forward(self, module, x=None):
    if x is None:
      x = range_like(self.input_spec, start=1)
    return module(x, is_training=False, test_local_stats=True)


@_register_golden(snt.BaseBatchNorm, "base_batch_norm_scale_offset_1x2x2x3")
class BaseBatchNormScaleOffset(AbstractGolden):
  create_module = (
      lambda _: snt.BaseBatchNorm(True, False, FooMetric(), FooMetric()))  # pytype: disable=wrong-arg-types
  input_spec = tf.TensorSpec([1, 2, 2, 3])
  num_variables = 2

  def forward(self, module, x=None):
    if x is None:
      x = range_like(self.input_spec, start=1)
    return module(x, is_training=False, test_local_stats=True)


@_register_golden(snt.BatchNorm, "batch_norm_1x2x2x3")
class BatchNorm(AbstractGolden):
  create_module = (lambda _: snt.BatchNorm(True, True))
  input_spec = tf.TensorSpec([1, 2, 2, 3])
  num_variables = 8

  def forward(self, module, x=None):
    if x is None:
      x = range_like(self.input_spec, start=1)
    return module(x, is_training=False, test_local_stats=True)


@_register_golden(snt.BatchNorm, "batch_norm_scale_offset_1x2x2x3")
class BatchNormScaleOffset(AbstractGolden):
  create_module = (lambda _: snt.BatchNorm(True, True))
  input_spec = tf.TensorSpec([1, 2, 2, 3])
  num_variables = 8

  def forward(self, module, x=None):
    if x is None:
      x = range_like(self.input_spec, start=1)
    return module(x, is_training=False, test_local_stats=True)


@_register_golden(snt.ExponentialMovingAverage, "ema_2")
class ExponentialMovingAverage(AbstractGolden):
  create_module = (lambda _: snt.ExponentialMovingAverage(decay=0.9))
  input_spec = tf.TensorSpec([2])
  num_variables = 3
  has_side_effects = True

  def forward(self, module, x=None):
    if x is None:
      x = range_like(self.input_spec, start=1)
    return module(x)


@_register_golden(snt.BatchNorm, "batch_norm_training_1x2x2x3")
class BatchNormTraining(AbstractGolden):
  create_module = (lambda _: snt.BatchNorm(True, True))
  input_spec = tf.TensorSpec([1, 2, 2, 3])
  num_variables = 8
  has_side_effects = True

  def forward(self, module, x=None):
    if x is None:
      x = range_like(self.input_spec, start=1)
    return module(x, is_training=True)


@_register_golden(snt.distribute.CrossReplicaBatchNorm,
                  "cross_replica_batch_norm_1x2x2x3")
class CrossReplicaBatchNorm(AbstractGolden):
  create_module = (
      lambda _: snt.BaseBatchNorm(True, False, FooMetric(), FooMetric()))
  input_spec = tf.TensorSpec([1, 2, 2, 3])
  num_variables = 2

  def forward(self, module, x=None):
    if x is None:
      x = range_like(self.input_spec, start=1)
    return module(x, is_training=False, test_local_stats=True)


@_register_golden(snt.Dropout, "dropout")
class DropoutVariableRate(AbstractGolden):
  create_module = lambda _: snt.Dropout(rate=tf.Variable(0.5))
  input_spec = tf.TensorSpec([3, 3, 3])
  num_variables = 1
  deterministic = False

  def forward(self, module, x=None):
    tf.random.set_seed(3)
    if x is None:
      x = range_like(self.input_spec, start=1)
    return module(x, is_training=True)


class AbstractRNNGolden(AbstractGolden):

  def forward(self, module, x=None):
    if x is None:
      # Small inputs to ensure that tf.tanh and tf.sigmoid don't saturate.
      x = 1.0 / range_like(self.input_spec, start=1)
    batch_size = self.input_spec.shape[0]
    prev_state = module.initial_state(batch_size)
    y, next_state = module(x, prev_state)
    del next_state
    return y


@_register_golden(snt.Conv1DLSTM, "conv1d_lstm_3x3_2x2")
class Conv1DLSTM(AbstractRNNGolden):
  input_spec = tf.TensorSpec([1, 2, 2])
  num_variables = 3

  def create_module(self):
    return snt.Conv1DLSTM(
        input_shape=self.input_spec.shape[1:],
        output_channels=3,
        kernel_shape=3)


@_register_golden(snt.Conv2DLSTM, "conv2d_lstm_3x3_2x2")
class Conv2DLSTM(AbstractRNNGolden):
  input_spec = tf.TensorSpec([1, 2, 2, 2])
  num_variables = 3

  def create_module(self):
    return snt.Conv2DLSTM(
        input_shape=self.input_spec.shape[1:],
        output_channels=3,
        kernel_shape=3)


@_register_golden(snt.Conv3DLSTM, "conv3d_lstm_3x3_2x2")
class Conv3DLSTM(AbstractRNNGolden):
  input_spec = tf.TensorSpec([1, 2, 2, 2, 2])
  num_variables = 3

  def create_module(self):
    return snt.Conv3DLSTM(
        input_shape=self.input_spec.shape[1:],
        output_channels=3,
        kernel_shape=3)


@_register_golden(snt.GRU, "gru_1")
class GRU(AbstractRNNGolden):
  create_module = lambda _: snt.GRU(hidden_size=1)
  input_spec = tf.TensorSpec([1, 128])
  num_variables = 3


@_register_golden(snt.LSTM, "lstm_1")
class LSTM(AbstractRNNGolden):
  create_module = lambda _: snt.LSTM(hidden_size=1)
  input_spec = tf.TensorSpec([1, 128])
  num_variables = 3


@_register_golden(snt.LSTM, "lstm_8_projected_1")
class LSTMWithProjection(AbstractRNNGolden):
  create_module = lambda _: snt.LSTM(hidden_size=8, projection_size=1)
  input_spec = tf.TensorSpec([1, 128])
  num_variables = 4


@_register_golden(snt.UnrolledLSTM, "unrolled_lstm_1")
class UnrolledLSTM(AbstractRNNGolden):
  create_module = lambda _: snt.UnrolledLSTM(hidden_size=1)
  input_spec = tf.TensorSpec([1, 1, 128])
  num_variables = 3


@_register_golden(snt.VanillaRNN, "vanilla_rnn_8")
class VanillaRNN(AbstractRNNGolden):
  create_module = lambda _: snt.VanillaRNN(hidden_size=8)
  input_spec = tf.TensorSpec([1, 128])
  num_variables = 3


@_register_golden(snt.TrainableState, "trainable_state")
class TrainableState(AbstractGolden):
  create_module = lambda _: snt.TrainableState(tf.zeros([1]))
  input_spec = tf.TensorSpec(())
  num_variables = 1


@_register_golden(snt.Bias, "bias_3x3x3")
class BiasTest(AbstractGolden):
  create_module = lambda _: snt.Bias()
  input_spec = tf.TensorSpec([1, 3, 3, 3])
  num_variables = 1


@_register_golden(snt.Embed, "embed_100_100")
class EmbedTest(AbstractGolden):
  create_module = lambda _: snt.Embed(vocab_size=100, embed_dim=100)
  input_spec = tf.TensorSpec([10], dtype=tf.int32)
  num_variables = 1


@_register_golden(snt.Mean, "mean_2x2")
class MeanTest(AbstractGolden):
  create_module = lambda _: snt.Mean()
  input_spec = tf.TensorSpec([2, 2])
  num_variables = 2
  has_side_effects = True


@_register_golden(snt.Sum, "sum_2x2")
class SumTest(AbstractGolden):
  create_module = lambda _: snt.Sum()
  input_spec = tf.TensorSpec([2, 2])
  num_variables = 1
  has_side_effects = True


@_register_golden(snt.nets.ResNet, "resnet50")
class ResNet(AbstractGolden):
  create_module = (lambda _: snt.nets.ResNet([1, 1, 1, 1], 9))
  input_spec = tf.TensorSpec([1, 8, 8, 3])
  num_variables = 155
  has_side_effects = True

  def forward(self, module, x=None):
    if x is None:
      x = range_like(self.input_spec, start=1)
    return module(x, is_training=True)


@_register_golden(snt.nets.VectorQuantizer, "vqvae")
class VectorQuantizerTest(AbstractGolden):

  def create_module(self):
    return snt.nets.VectorQuantizer(
        embedding_dim=4, num_embeddings=6, commitment_cost=0.25)

  # Input can be any shape as long as final dimension is equal to embedding_dim.
  input_spec = tf.TensorSpec([2, 3, 4])

  def forward(self, module, x=None):
    if x is None:
      x = range_like(self.input_spec)
    return module(x, is_training=True)

  # Numerical results can be quite different on TPU, be a bit more loose here.
  tpu_atol = 4e-2

  num_variables = 1


@_register_golden(snt.nets.VectorQuantizerEMA, "vqvae_ema_train")
class VectorQuantizerEMATrainTest(AbstractGolden):

  def create_module(self):
    return snt.nets.VectorQuantizerEMA(
        embedding_dim=5, num_embeddings=7, commitment_cost=0.5, decay=0.9)

  # Input can be any shape as long as final dimension is equal to embedding_dim.
  input_spec = tf.TensorSpec([2, 5])

  def forward(self, module, x=None):
    if x is None:
      x = range_like(self.input_spec)
    return module(x, is_training=True)

  # Numerical results can be quite different on TPU, be a bit more loose here.
  tpu_atol = 4e-2

  num_variables = 7  # 1 embedding, then 2 EMAs each of which contain 3.
  has_side_effects = True


@_register_golden(snt.nets.VectorQuantizerEMA, "vqvae_ema_eval")
class VectorQuantizerEMAEvalTest(AbstractGolden):

  def create_module(self):
    return snt.nets.VectorQuantizerEMA(
        embedding_dim=3, num_embeddings=4, commitment_cost=0.5, decay=0.9)

  # Input can be any shape as long as final dimension is equal to embedding_dim.
  input_spec = tf.TensorSpec([2, 3])

  def forward(self, module, x=None):
    if x is None:
      x = range_like(self.input_spec)
    return module(x, is_training=False)

  # Numerical results can be quite different on TPU, be a bit more loose here.
  tpu_atol = 4e-2

  num_variables = 7  # 1 embedding, then 2 EMAs each of which contain 3.
  has_side_effects = False  # only has side effects when is_training==True


# pylint: enable=missing-docstring


class FooMetric(snt.Metric):
  """Used for testing a class which uses Metrics."""

  def initialize(self, x):
    pass

  def reset(self):
    pass

  def update(self, x):
    pass
