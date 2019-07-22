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

"""Ensures that all Sonnet modules support ``tf.function``."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import collections

from absl.testing import parameterized
import sonnet as snt
from sonnet.src import test_utils
import tensorflow as tf
from typing import Callable, Tuple


class Wrapped(snt.Module):

  @snt.no_name_scope
  def __init__(self, wrapped: snt.Module):
    super(Wrapped, self).__init__()
    self.wrapped = wrapped


class Training(Wrapped):

  @snt.no_name_scope
  def __call__(self, x: tf.Tensor):
    return self.wrapped(x, is_training=True)


class Core(Wrapped):

  @snt.no_name_scope
  def __call__(self, x: tf.Tensor):
    state = self.wrapped.initial_state(batch_size=tf.shape(x)[0])
    return self.wrapped(x, state)


# TODO(tomhennigan) De-duplicate this, BATCH_MODULES and goldens.py.
ModuleDescriptor = collections.namedtuple(
    "ModuleDescriptor", ["name", "create", "shape", "dtype"])
ModuleDescriptor.__new__.__defaults__ = (None, None, None, tf.float32)
ModuleFn = Callable[[], snt.Module]

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
        name="Conv1DLSTM",
        create=lambda: Core(snt.Conv1DLSTM((2, 2), 3, 3)),
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
        name="Conv2DLSTM",
        create=lambda: Core(snt.Conv2DLSTM((2, 2, 2), 3, 3)),
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
        name="Conv3DLSTM",
        create=lambda: Core(snt.Conv3DLSTM((2, 2, 2, 2), 3, 3)),
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
        name="GRU",
        create=lambda: Core(snt.GRU(1)),
        shape=(BATCH_SIZE, 128)),
    ModuleDescriptor(
        name="GroupNorm",
        create=lambda: snt.GroupNorm(2, True, True),
        shape=(BATCH_SIZE, 3, 4)),
    ModuleDescriptor(
        name="InstanceNorm",
        create=lambda: snt.InstanceNorm(True, True),
        shape=(BATCH_SIZE, 3, 2)),
    ModuleDescriptor(
        name="LSTM",
        create=lambda: Core(snt.LSTM(1)),
        shape=(BATCH_SIZE, 128)),
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
        name="VanillaRNN",
        create=lambda: Core(snt.VanillaRNN(8)),
        shape=(BATCH_SIZE, 128)),
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
        name="nets.MLP",
        create=lambda: snt.nets.MLP([3, 4, 5]),
        shape=(BATCH_SIZE, 3)),
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
    snt.DeepRNN, snt.RNNCore, snt.TrainableState,
}


def unwrap(module: snt.Module) -> snt.Module:
  while isinstance(module, Wrapped):
    module = module.wrapped
  return module


class FunctionTest(test_utils.TestCase, parameterized.TestCase):

  def test_coverage(self):
    all_modules = frozenset(test_utils.find_all_sonnet_modules(snt, snt.Module))
    tested_modules = {type(unwrap(d.create()))
                      for d in BATCH_MODULES + OPTIMIZER_MODULES}
    self.assertEmpty(all_modules - (tested_modules | IGNORED_MODULES))

  @test_utils.combined_named_parameters(BATCH_MODULES,
                                        test_utils.named_bools("autograph"))
  def test_trace(
      self,
      module_fn: ModuleFn,
      input_shape: Tuple[int],
      dtype: tf.DType,
      autograph: bool,
  ):
    module = module_fn()
    forward = tf.function(module, autograph=autograph)
    forward(tf.ones(input_shape, dtype=dtype))

  @test_utils.combined_named_parameters(BATCH_MODULES,
                                        test_utils.named_bools("autograph"))
  def test_trace_batch_agnostic(
      self,
      module_fn: ModuleFn,
      input_shape: Tuple[int],
      dtype: tf.DType,
      autograph: bool,
  ):
    module = module_fn()
    forward = tf.function(module, autograph=autograph)
    input_spec = tf.TensorSpec((None,) + input_shape[1:], dtype=dtype)
    cf = forward.get_concrete_function(input_spec)
    cf(tf.ones(input_shape, dtype=dtype))

  @test_utils.combined_named_parameters(BATCH_MODULES,
                                        test_utils.named_bools("autograph"))
  def test_trace_batch_apply_batch_agnostic(
      self,
      module_fn: ModuleFn,
      input_shape: Tuple[int],
      dtype: tf.DType,
      autograph: bool,
  ):
    module = snt.BatchApply(module_fn())
    forward = tf.function(module, autograph=autograph)
    input_shape = (8,) + input_shape
    input_spec = tf.TensorSpec((None, None) + input_shape[2:], dtype=dtype)
    cf = forward.get_concrete_function(input_spec)
    if isinstance(unwrap(module.module), (snt.nets.VectorQuantizer,
                                          snt.nets.VectorQuantizerEMA)):
      # TODO(tomhennigan) Make VQ and VQ-EMA batch agnostic under BatchApply.
      return
    cf(tf.ones(input_shape, dtype=dtype))

  @test_utils.combined_named_parameters(OPTIMIZER_MODULES,
                                        test_utils.named_bools("autograph"))
  def test_optimizer_dense(
      self,
      optimizer_fn: ModuleFn,
      input_shape: Tuple[int],
      dtype: tf.DType,
      autograph: bool,
  ):
    del input_shape, dtype  # Unused.
    parameters = [tf.Variable([1., 2.])]
    updates = [tf.constant([5., 5.])]
    optimizer = optimizer_fn()
    optimizer_apply = tf.function(optimizer.apply, autograph=autograph)
    optimizer_apply(updates, parameters)

  # TODO(petebu) Add a test with completely dynamic shapes.

  @test_utils.combined_named_parameters(OPTIMIZER_MODULES,
                                        test_utils.named_bools("autograph"))
  def test_optimizer_sparse(
      self,
      optimizer_fn: ModuleFn,
      input_shape: Tuple[int],
      dtype: tf.DType,
      autograph: bool,
  ):
    del input_shape, dtype  # Unused.
    if self.primary_device == "TPU":
      self.skipTest("IndexedSlices not supported on TPU.")
    parameters = [tf.Variable([[1.], [2.]])]
    updates = [tf.IndexedSlices(tf.constant([0.1], shape=[1, 1]),
                                tf.constant([0]), tf.constant([2, 1]))]
    optimizer = optimizer_fn()
    optimizer_apply = tf.function(optimizer.apply, autograph=autograph)
    optimizer_apply(updates, parameters)


if __name__ == "__main__":
  # tf.enable_v2_behavior()
  tf.test.main()
