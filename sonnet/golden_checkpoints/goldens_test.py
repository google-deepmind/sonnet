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

"""Tests for sonnet.v2.golden_checkpoints."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import os
import pickle

from absl.testing import absltest
from absl.testing import parameterized
import sonnet as snt
from sonnet.golden_checkpoints import goldens
from sonnet.src import test_utils
import tensorflow as tf


class TestCheckpoint(object):
  """Wraps a tf.train.Checkpoint to make it more convenient for testing."""

  def __init__(self, root=None, **kwargs):
    if root is None:
      root = absltest.get_default_test_tmpdir()
    self._root = root
    self._prefix = os.path.join(self._root, "checkpoint")
    self._checkpoint = tf.train.Checkpoint(**kwargs)

  def save(self):
    self._checkpoint.save(file_prefix=self._prefix)

  def restore_latest(self, assert_consumed=True):
    status = self._checkpoint.restore(tf.train.latest_checkpoint(self._root))
    if assert_consumed:
      # Ensures that all values in the checkpoint have been consumed by some
      # checkpointable Python object.
      status.assert_consumed()
    return status


def all_goldens(test_method):
  cases = ((name, cls()) for _, name, cls in goldens.list_goldens())
  return parameterized.named_parameters(cases)(test_method)


def mirrored_all_devices():
  # NOTE: We avoid the default constructor so we mirror over CPU, CPU + GPU and
  # all TPU cores (the default ctor currently does all CPUs or all GPUs).
  all_visible_devices = [
      d.name for d in tf.config.experimental.list_logical_devices()
      if d.device_type != "TPU_SYSTEM"]
  return tf.distribute.MirroredStrategy(devices=all_visible_devices)


def all_goldens_and_strategies(test_method):
  # TODO(tomhennigan) Add TPU and ParameterServer tests.
  cases = [(name + "_mirrored", cls(), mirrored_all_devices)
           for _, name, cls in goldens.list_goldens()]
  return parameterized.named_parameters(cases)(test_method)


class GoldenCheckpointsTest(test_utils.TestCase, parameterized.TestCase):
  """Adds test methods running standard checkpointing tests."""

  @all_goldens
  def test_save_load(self, golden):
    """Test a basic save/load cycle."""
    module = golden.create_module()
    checkpoint = TestCheckpoint(module=module)
    all_variables = golden.create_all_variables(module)

    # Save zeros into the checkpoint.
    self.assertNotEmpty(all_variables)
    self.assertEqual(set(all_variables), set(module.variables))
    for variable in all_variables:
      # TODO(tomhennigan) Perhaps limit the range/switch to random to avoid
      # overflow/underflow in the forward pass?
      variable.assign(goldens.range_like(variable))
    old_y = golden.forward(module)
    checkpoint.save()

    # Overwrite zeros with ones.
    for variable in all_variables:
      variable.assign(tf.ones_like(variable))

    # Check restored values match the saved values.
    checkpoint.restore_latest()
    for variable in all_variables:
      self.assertAllClose(variable.read_value(), goldens.range_like(variable))

    # Test the output from the module remains stable.
    # TODO(tomhennigan) Handle modules with nested outputs.
    if golden.deterministic:
      self.assertAllClose(golden.forward(module), old_y)

  @all_goldens
  def test_save_then_load_new_instance(self, golden):
    """Checks that a checkpoint created for one instance can restore another."""
    module_1 = golden.create_module()
    checkpoint_1 = TestCheckpoint(module=module_1)
    variables_1 = golden.create_all_variables(module_1)

    module_2 = golden.create_module()
    checkpoint_2 = TestCheckpoint(module=module_2)
    variables_2 = golden.create_all_variables(module_2)

    for v1, v2 in zip(variables_1, variables_2):
      v1.assign(goldens.range_like(v1))
      v2.assign(tf.ones_like(v2))

    checkpoint_1.save()
    checkpoint_2.restore_latest()

    # Assert the parameters in both modules are the same.
    for variable in variables_2:
      self.assertAllClose(variable.read_value(), goldens.range_like(variable))

    # Assert the output from both modules are the same.
    # TODO(tomhennigan) Handle modules with nested outputs.
    if golden.deterministic:
      self.assertAllClose(golden.forward(module_1), golden.forward(module_2))

  @all_goldens
  def test_restore_on_create(self, golden):
    """Tests that Variable values are restored on creation."""
    # Create a module, set its variables to sequential values and save.
    module_1 = golden.create_module()
    checkpoint_1 = TestCheckpoint(module=module_1)
    variables_1 = golden.create_all_variables(module_1)
    for variable in variables_1:
      variable.assign(goldens.range_like(variable))
    checkpoint_1.save()

    # Create a different module, restore from a checkpoint, create parameters
    # and assert their values are sequential.
    module_2 = golden.create_module()
    checkpoint_2 = TestCheckpoint(module=module_2)
    status = checkpoint_2.restore_latest(assert_consumed=False)
    variables_2 = golden.create_all_variables(module_2)
    status.assert_consumed()
    for variable in variables_2:
      self.assertAllClose(variable.read_value(), goldens.range_like(variable))

    # Assert the output from both modules is the same.
    # TODO(tomhennigan) Handle modules with nested outputs.
    if golden.deterministic:
      self.assertAllClose(golden.forward(module_1), golden.forward(module_2))

  @all_goldens
  def test_restore_golden(self, golden):
    """Test restoring from a golden checkpoint still works."""
    module = golden.create_module()
    root = os.path.join(
        "sonnet/golden_checkpoints/",
        golden.name)
    checkpoint = TestCheckpoint(root=root, module=module)
    variables = golden.create_all_variables(module)
    for variable in variables:
      variable.assign(tf.zeros_like(variable))
    checkpoint.restore_latest()
    for variable in variables:
      self.assertAllClose(variable.read_value(), goldens.range_like(variable))

  @all_goldens_and_strategies
  def test_checkpoint_distribution_strategy(self, golden, strategy_fn):
    strategy = strategy_fn()
    with strategy.scope():
      module = golden.create_module()
      variables = golden.create_all_variables(module)

    def forward():
      per_replica = strategy.experimental_run_v2(lambda: golden.forward(module))
      return tf.stack(strategy.unwrap(per_replica), axis=0)

    # Assign sequential values to the weights and compute a forward pass.
    for index, variable in enumerate(variables):
      variable.assign(goldens.range_like(variable, start=index))
    before_save_ys = forward()

    # Create a checkpoint and save the weights.
    checkpoint = TestCheckpoint(module=module)
    checkpoint.save()

    # Assign ones into the weights and do another forward pass. The result
    # should be different.
    for variable in variables:
      variable.assign(tf.ones_like(variable))

    if golden.deterministic:
      y = forward()
      self.assertNotAllClose(y, before_save_ys)

    # Restore from the checkpoint and assert the module is in the same state.
    checkpoint.restore_latest()

    for index, variable in enumerate(variables):
      # Parameters should be restored to their previous values.
      self.assertAllEqual(variable.read_value(),
                          goldens.range_like(variable, start=index))

    if golden.deterministic:
      self.assertAllEqual(forward(), before_save_ys)


class SavedModelTest(test_utils.TestCase, parameterized.TestCase):

  @all_goldens
  def test_save_restore_cycle(self, golden):
    module = golden.create_module()

    # Create all parameters and set them to sequential (but different) values.
    variables = golden.create_all_variables(module)
    for index, variable in enumerate(variables):
      variable.assign(goldens.range_like(variable, start=index))

    @tf.function(input_signature=[golden.input_spec])
    def inference(x):
      # We'll let `golden.forward` run the model with a fixed input. This allows
      # for additional positional arguments like is_training.
      del x
      return golden.forward(module)

    # Create a saved model, add a method for inference and a dependency on our
    # module such that it can find dependencies.
    saved_model = snt.Module()
    saved_model._module = module
    saved_model.inference = inference
    saved_model.all_variables = list(module.variables)

    # Sample input, the value is not important (it is not used in the inference
    # function).
    x = goldens.range_like(golden.input_spec)

    # Run the saved model and pull variable values.
    y1 = saved_model.inference(x)
    v1 = saved_model.all_variables

    # Save the model to disk and restore it.
    tmp_dir = os.path.join(absltest.get_default_test_tmpdir(), golden.name)
    tf.saved_model.save(saved_model, tmp_dir)
    restored_model = tf.saved_model.load(tmp_dir)

    # Run the loaded model and pull variable values.
    y2 = restored_model.inference(x)
    v2 = restored_model.all_variables

    if golden.deterministic:
      # The output before and after saving should be exactly the same.
      self.assertAllEqual(y1, y2)

    for a, b in zip(v1, v2):
      self.assertEqual(a.name, b.name)
      self.assertEqual(a.device, b.device)
      self.assertAllEqual(a.read_value(), b.read_value())


class PickleTest(test_utils.TestCase, parameterized.TestCase):

  # TODO(tomhennigan) Add tests with dill and cloudpickle.

  @all_goldens
  def test_pickle(self, golden):
    m1 = golden.create_module()
    y1 = golden.forward(m1)
    m2 = pickle.loads(pickle.dumps(m1))
    for v1, v2 in zip(m1.variables, m2.variables):
      self.assertAllEqual(v1.read_value(), v2.read_value())
    if golden.deterministic:
      y2 = golden.forward(m2)
      self.assertAllEqual(y1, y2)


class CoverageTest(test_utils.TestCase):

  def test_all_modules_covered(self):
    no_checkpoint_whitelist = set([
        # TODO(petebu): Remove this once optimizer goldens check works.
        snt.optimizers.Adam,
        snt.optimizers.Momentum,
        snt.optimizers.RMSProp,
        snt.optimizers.SGD,

        # Stateless or abstract.
        snt.Module,
        snt.DeepRNN,
        snt.Deferred,
        snt.Flatten,
        snt.Metric,
        snt.Reshape,
        snt.RNNCore,
        snt.Sequential,
        snt.src.recurrent._ConvNDLSTM,
        snt.src.recurrent._LegacyDeepRNN,
        snt.src.recurrent._RecurrentDropoutWrapper,
        snt.src.recurrent._ResidualWrapper,

        # TODO(slebedev): remove these once CuDNN cores are exported.
        snt.src.recurrent.CuDNNGRU,
        snt.src.recurrent.CuDNNLSTM,

        # TODO(tamaranorman) remove these when visibility set correctly
        snt.src.conv.ConvND,
        snt.src.conv_transpose.ConvNDTranspose,
        snt.src.moving_averages.ExponentialMovingAverage,
        snt.src.adam.ReferenceAdam,
        snt.src.momentum.ReferenceMomentum,
        snt.src.rmsprop.ReferenceRMSProp,
        snt.src.sgd.ReferenceSGD,
    ])

    # Find all the snt.Module types reachable from `import sonnet as snt`
    all_sonnet_types = set()
    for _, python_module in test_utils.find_sonnet_python_modules(snt):
      for _, cls in inspect.getmembers(python_module, inspect.isclass):
        if issubclass(cls, snt.Module):
          all_sonnet_types.add(cls)

    # Find all the modules that have checkpoint tests.
    tested_modules = {module_cls for module_cls, _, _ in goldens.list_goldens()}

    # Make sure we don't leave entries in no_checkpoint_whitelist if they are
    # actually tested.
    self.assertEmpty(tested_modules & no_checkpoint_whitelist)

    # Make sure everything is covered.
    self.assertEqual(tested_modules | no_checkpoint_whitelist, all_sonnet_types)


if __name__ == "__main__":
  # tf.enable_v2_behavior()
  tf.test.main()
