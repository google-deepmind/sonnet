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
"""Module descriptors programatically describe how to use modules."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import collections

import sonnet as snt
import tensorflow as tf


class Wrapped(snt.Module):

  @snt.no_name_scope
  def __init__(self, wrapped: snt.Module):
    super(Wrapped, self).__init__()
    self.wrapped = wrapped


class Training(Wrapped):

  @snt.no_name_scope
  def __call__(self, x: tf.Tensor):
    return self.wrapped(x, is_training=True)


class Recurrent(Wrapped):

  @snt.no_name_scope
  def __call__(self, x: tf.Tensor):
    initial_state = self.wrapped.initial_state(batch_size=tf.shape(x)[0])
    if isinstance(self.wrapped, snt.UnrolledRNN):
      # The module expects TB...-shaped input as opposed to BT...
      x = tf.transpose(x, [1, 0] +  list(range(2, x.shape.rank)))
    return self.wrapped(x, initial_state)


def unwrap(module: snt.Module) -> snt.Module:
  while isinstance(module, Wrapped):
    module = module.wrapped
  return module


# TODO(tomhennigan) De-duplicate this, BATCH_MODULES and goldens.py.
ModuleDescriptor = collections.namedtuple(
    "ModuleDescriptor", ["name", "create", "shape", "dtype"])
ModuleDescriptor.__new__.__defaults__ = (None, None, None, tf.float32)

BATCH_SIZE = 8

# pylint: disable=unnecessary-lambda
BATCH_MODULES = (
    ModuleDescriptor(
        name="BatchNorm",
        create=lambda: Training(snt.BatchNorm(True, True)),
        shape=(BATCH_SIZE, 2, 2, 3)),
    ModuleDescriptor(
        name="Bias",
        create=lambda: snt.Bias(),
        shape=(BATCH_SIZE, 3, 3, 3)),
    ModuleDescriptor(
        name="Conv1D",
        create=lambda: snt.Conv1D(3, 3),
        shape=(BATCH_SIZE, 2, 2)),
    ModuleDescriptor(
        name="Conv1DTranspose",
        create=lambda: snt.Conv1DTranspose(3, 3),
        shape=(BATCH_SIZE, 2, 2)),
    ModuleDescriptor(
        name="Conv2D",
        create=lambda: snt.Conv2D(3, 3),
        shape=(BATCH_SIZE, 2, 2, 2)),
    ModuleDescriptor(
        name="Conv2DTranspose",
        create=lambda: snt.Conv2DTranspose(3, 3),
        shape=(BATCH_SIZE, 2, 2, 2)),
    ModuleDescriptor(
        name="Conv3D",
        create=lambda: snt.Conv3D(3, 3),
        shape=(BATCH_SIZE, 2, 2, 2, 2)),
    ModuleDescriptor(
        name="Conv3DTranspose",
        create=lambda: snt.Conv3DTranspose(3, 3),
        shape=(BATCH_SIZE, 2, 2, 2, 2)),
    ModuleDescriptor(
        name="Dropout",
        create=lambda: Training(snt.Dropout(0.5)),
        shape=(BATCH_SIZE, 3, 3)),
    ModuleDescriptor(
        name="Embed",
        create=lambda: snt.Embed(10),
        shape=(BATCH_SIZE,),
        dtype=tf.int32),
    ModuleDescriptor(
        name="Flatten",
        create=lambda: snt.Flatten(),
        shape=(BATCH_SIZE, 3, 3, 3)),
    ModuleDescriptor(
        name="GroupNorm",
        create=lambda: snt.GroupNorm(2, True, True),
        shape=(BATCH_SIZE, 3, 4)),
    ModuleDescriptor(
        name="InstanceNorm",
        create=lambda: snt.InstanceNorm(True, True),
        shape=(BATCH_SIZE, 3, 2)),
    ModuleDescriptor(
        name="LayerNorm",
        create=lambda: snt.LayerNorm(1, True, True),
        shape=(BATCH_SIZE, 3, 2)),
    ModuleDescriptor(
        name="Linear",
        create=lambda: snt.Linear(10),
        shape=(BATCH_SIZE, 1)),
    ModuleDescriptor(
        name="Sequential",
        create=lambda: snt.Sequential([lambda x: x]),
        shape=(BATCH_SIZE, 2, 2)),
    ModuleDescriptor(
        name="nets.VectorQuantizer",
        create=lambda: Training(snt.nets.VectorQuantizer(4, 6, 0.25)),
        shape=(BATCH_SIZE, 3, 4)),
    ModuleDescriptor(
        name="nets.VectorQuantizerEMA",
        create=lambda: Training(snt.nets.VectorQuantizerEMA(5, 7, 0.5, 0.9)),
        shape=(BATCH_SIZE, 5)),
    ModuleDescriptor(
        name="nets.Cifar10ConvNet",
        create=lambda: Training(snt.nets.Cifar10ConvNet()),
        shape=(BATCH_SIZE, 3, 3, 2)),
    ModuleDescriptor(
        name="nets.ResNet50",
        create=lambda: Training(snt.nets.ResNet([1, 1, 1, 1], 4)),
        shape=(BATCH_SIZE, 3, 3, 2)),
    ModuleDescriptor(
        name="nets.MLP",
        create=lambda: snt.nets.MLP([3, 4, 5]),
        shape=(BATCH_SIZE, 3)),
)

RECURRENT_MODULES = (
    ModuleDescriptor(
        name="Conv1DLSTM",
        create=lambda: Recurrent(snt.Conv1DLSTM((2, 2), 3, 3)),
        shape=(BATCH_SIZE, 2, 2)),
    ModuleDescriptor(
        name="Conv2DLSTM",
        create=lambda: Recurrent(snt.Conv2DLSTM((2, 2, 2), 3, 3)),
        shape=(BATCH_SIZE, 2, 2, 2)),
    ModuleDescriptor(
        name="Conv3DLSTM",
        create=lambda: Recurrent(snt.Conv3DLSTM((2, 2, 2, 2), 3, 3)),
        shape=(BATCH_SIZE, 2, 2, 2, 2)),
    ModuleDescriptor(
        name="GRU",
        create=lambda: Recurrent(snt.GRU(1)),
        shape=(BATCH_SIZE, 128)),
    ModuleDescriptor(
        name="LSTM",
        create=lambda: Recurrent(snt.LSTM(1)),
        shape=(BATCH_SIZE, 128)),
    ModuleDescriptor(
        name="UnrolledLSTM",
        create=lambda: Recurrent(snt.UnrolledLSTM(1)),
        shape=(BATCH_SIZE, 1, 128)),
    ModuleDescriptor(
        name="VanillaRNN",
        create=lambda: Recurrent(snt.VanillaRNN(8)),
        shape=(BATCH_SIZE, 128)),
)

OPTIMIZER_MODULES = (
    ModuleDescriptor(
        name="optimizers.Adam",
        create=lambda: snt.optimizers.Adam(learning_rate=0.1)),
    ModuleDescriptor(
        name="optimizers.Momentum",
        create=lambda: snt.optimizers.Momentum(learning_rate=0.1, momentum=.9)),
    ModuleDescriptor(
        name="optimizers.RMSProp",
        create=lambda: snt.optimizers.RMSProp(learning_rate=0.1)),
    ModuleDescriptor(
        name="optimizers.SGD",
        create=lambda: snt.optimizers.SGD(learning_rate=0.1)),
)

IGNORED_MODULES = {
    # Stateless or abstract.
    snt.BatchApply, snt.Deferred, snt.Module, snt.Optimizer, snt.Reshape,

    # Metrics.
    snt.ExponentialMovingAverage, snt.Mean, snt.Metric, snt.Sum,

    # Normalization.
    snt.BaseBatchNorm,  # Tested via `snt.BatchNorm`.

    # Recurrent.
    snt.DeepRNN, snt.RNNCore, snt.TrainableState, snt.UnrolledRNN,

    # Nets,
    snt.nets.ResNet50,  # Tested via `snt.nets.ResNet`.
}
