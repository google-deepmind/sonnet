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

from typing import Callable, Tuple

from absl.testing import parameterized
import sonnet as snt
from sonnet.src import test_utils
from sonnet.src.conformance import descriptors
import tensorflow as tf

ModuleFn = Callable[[], snt.Module]
BATCH_MODULES = descriptors.BATCH_MODULES
RECURRENT_MODULES = descriptors.RECURRENT_MODULES
OPTIMIZER_MODULES = descriptors.OPTIMIZER_MODULES
IGNORED_MODULES = descriptors.IGNORED_MODULES


class FunctionTest(test_utils.TestCase, parameterized.TestCase):

  @test_utils.combined_named_parameters(BATCH_MODULES + RECURRENT_MODULES,
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

  @test_utils.combined_named_parameters(BATCH_MODULES + RECURRENT_MODULES,
                                        test_utils.named_bools("autograph"))
  def test_create_variables_eagerly(
      self,
      module_fn: ModuleFn,
      input_shape: Tuple[int],
      dtype: tf.DType,
      autograph: bool,
  ):
    module = module_fn()
    f = snt.distribute.create_variables_eagerly(module)
    forward = tf.function(f, autograph=autograph)
    forward(tf.ones(input_shape, dtype=dtype))

  @test_utils.combined_named_parameters(BATCH_MODULES + RECURRENT_MODULES,
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
    if isinstance(
        descriptors.unwrap(module.module),
        (snt.nets.VectorQuantizer, snt.nets.VectorQuantizerEMA)):
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
    updates = [
        tf.IndexedSlices(
            tf.constant([0.1], shape=[1, 1]), tf.constant([0]),
            tf.constant([2, 1]))
    ]
    optimizer = optimizer_fn()
    optimizer_apply = tf.function(optimizer.apply, autograph=autograph)
    optimizer_apply(updates, parameters)


if __name__ == "__main__":
  tf.test.main()
