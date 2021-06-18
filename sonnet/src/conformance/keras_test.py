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
"""Tests Sonnet and Keras compatibility."""

from absl.testing import parameterized
import sonnet as snt
from sonnet.src import test_utils
from sonnet.src.conformance import descriptors
import tensorflow as tf
import tree

BATCH_MODULES = descriptors.BATCH_MODULES
RECURRENT_MODULES = descriptors.RECURRENT_MODULES


# TODO(tomhennigan) Add tests with Keras optimizers.
# TODO(tomhennigan) Test Keras compile/fit.
class KerasTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(*(BATCH_MODULES + RECURRENT_MODULES))
  def test_build_without_batch(self, module_fn, input_shape, dtype):
    # For Keras test that building with unknown batch dim is supported.
    layer = LayerAdapter(module=module_fn(), dtype=dtype)
    layer.build((None,) + input_shape[1:])

    # For Sonnet just call with the example input.
    mod = module_fn()
    mod(tf.ones(input_shape, dtype))

    # Some modules (e.g. Sequential) are parameter-less.
    snt.allow_empty_variables(mod)

    # Test that module variables look the same.
    by_name = lambda c: sorted(c, key=lambda v: v.name)
    abstract = lambda v: (v.name, v.shape, v.dtype)
    for collection in ("variables", "trainable_variables"):
      for m, l in zip(
          by_name(getattr(mod, collection)),
          by_name(getattr(layer, collection))):
        self.assertEqual(abstract(m), abstract(l))

  @parameterized.named_parameters(*(BATCH_MODULES + RECURRENT_MODULES))
  def test_sonnet_module_as_layer(self, module_fn, input_shape, dtype):
    mod = module_fn()
    layer = LayerAdapter(module=module_fn(), dtype=dtype)
    example_input = tf.ones(input_shape, dtype=dtype)

    # Check outputs are the same.
    for m_y, l_y in zip(
        tree.flatten(mod(example_input)), tree.flatten(layer(example_input))):
      self.assertEqual(m_y.shape, l_y.shape)
      self.assertEqual(m_y.dtype, l_y.dtype)

    # Some modules (e.g. Sequential) are parameter-less.
    snt.allow_empty_variables(mod)

    # Check that variables are the same.
    self.assertEqual(len(mod.variables), len(layer.variables))
    self.assertEqual(
        len(mod.trainable_variables), len(layer.trainable_variables))

    # Check that Keras layer freezing works
    layer.trainable = False
    self.assertEmpty(layer.trainable_variables)

  def test_build_with_updating_module(self):
    # Calling the module creates and updates `w`.
    mod = ModuleWithUpdateInCall()
    mod(tf.ones([]))
    self.assertEqual(mod.w.numpy(), 1)

    # Calling build() should not trigger updating `w`, just creating it.
    layer = LayerAdapter(ModuleWithUpdateInCall())
    layer.build([])
    self.assertEqual(layer.module.w.numpy(), 0)

  def test_layer_with_model(self):
    layers = [
        LayerAdapter(snt.Linear(3)),
        LayerAdapter(snt.Linear(2)),
        LayerAdapter(snt.Linear(1))
    ]

    model = tf.keras.models.Sequential(layers)
    model.build([None, 4])
    for idx, input_size in enumerate([4, 3, 2]):
      self.assertEqual(layers[idx].module.input_size, input_size)

    output_shape = model.compute_output_shape([None, 4])
    self.assertTrue(output_shape.is_compatible_with([None, 1]))

    self.assertEqual(model(tf.ones([1, 4])).shape, [1, 1])

  @parameterized.named_parameters(*(BATCH_MODULES + RECURRENT_MODULES))
  def test_symbolic_model(self, module_fn, input_shape, dtype):
    module = module_fn()

    inputs = tf.keras.Input(input_shape[1:], dtype=dtype)
    layer = LayerAdapter(module=module, dtype=dtype)
    output = layer(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=output)

    example_input = tf.ones(input_shape, dtype=dtype)
    # Check outputs are the same.
    for m_y, l_y in zip(
        tree.flatten(module(example_input)),
        tree.flatten(model(example_input))):
      self.assertEqual(m_y.shape, l_y.shape)
      self.assertEqual(m_y.dtype, l_y.dtype)

  def test_layer_adapter_custom_method(self):
    module = ModuleWithCustomForward()

    inputs = tf.keras.Input([], batch_size=1)
    layer = LayerAdapter(module=module, method="forward")
    output = layer(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=output)

    self.assertEqual(model(tf.ones([])).numpy(), [2.])
    self.assertEqual(model.trainable_variables, [module.w])

  def test_keras_layer_inside_sonnet_module(self):
    mod = ModuleWithLayer()
    mod(tf.ones([1, 1]))
    self.assertEqual(mod.submodules, (mod.dense,))
    self.assertLen(mod.variables, 2)
    self.assertLen(mod.trainable_variables, 2)

    # Test that layer freezing does not change tf.Module tracking.
    mod.dense.trainable = False
    self.assertLen(mod.variables, 2)
    self.assertLen(mod.trainable_variables, 2)

  def test_to_config(self):
    mod = LayerAdapter(ModuleWithLayer())
    with self.assertRaises(NotImplementedError):
      mod.to_config()

  def test_from_config(self):
    with self.assertRaises(NotImplementedError):
      LayerAdapter.from_config(None)


# TODO(tomhennigan) Make this part of the public API?
class LayerAdapter(tf.keras.layers.Layer):
  """Adapts a Sonnet module to conform to the Keras Layer API.

      >>> layer = LayerAdapter(snt.Linear(1))
      >>> assert isinstance(layer, tf.keras.layers.Layer)

  We support building without ``__call__``, even with unknown dimensions:

      >>> layer.build(input_shape=[None, 28 * 28])

  Of course now features of Keras work as expected, for example layer freezing:

      >>> [v.name for v in layer.trainable_variables]
      ["linear/b:0", "linear/w:0"]

      >>> layer.trainable = False
      >>> layer.trainable_variables
      []
  """

  def __init__(self, module, method="__call__", dtype=tf.float32):
    super().__init__(dtype=dtype)
    self.module = module
    self._module_call_method = getattr(module, method)
    self._output_shapes = None

  @classmethod
  def from_config(cls, config):
    raise NotImplementedError

  def to_config(self):
    raise NotImplementedError

  def _trace_and_initialize(self, input_shape):
    if self._output_shapes is None:
      self._output_shapes = tree.map_structure(
          lambda spec: spec.shape if spec is not None else spec,
          snt.build(self, tf.TensorSpec(input_shape, self.dtype)))

    return self._output_shapes

  def compute_output_shape(self, input_shape):
    output_shapes = self._trace_and_initialize(input_shape)
    return output_shapes

  def build(self, input_shape):
    super().build(input_shape)

    # Trigger variable initialization by tracing the module.
    self._trace_and_initialize(input_shape)

    # Make sure Keras variable tracking finds our weights.
    # Keras has a setattr override which can be used to register weights in a
    # similar way to `Layer.add_weight`. By setting `_sonnet_weights` we trigger
    # this mechanism and module weights are found in `Layer.trainable_variables`
    # and `Layer.variables`.
    snt.allow_empty_variables(self.module)
    self._sonnet_weights = self.module.variables

  def call(self, inputs):
    return self._module_call_method(inputs)


class ModuleWithLayer(snt.Module):

  def __init__(self):
    super().__init__()
    self.dense = tf.keras.layers.Dense(10)

  def __call__(self, x):
    return self.dense(x)


class ModuleWithUpdateInCall(snt.Module):

  @snt.once
  def _init(self, x):
    self.w = tf.Variable(tf.zeros(x.shape), name="w")

  def __call__(self, x):
    self._init(x)
    self.w.assign_add(tf.ones_like(self.w))
    return self.w.read_value()


class ModuleWithCustomForward(snt.Module):

  @snt.once
  def _init(self, x):
    self.w = tf.Variable(tf.ones(x.shape), name="w")

  def forward(self, x):
    self._init(x)
    return x + self.w


if __name__ == "__main__":
  tf.test.main()
