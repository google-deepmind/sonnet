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

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import sonnet as snt
from sonnet.golden_checkpoints import goldens
from sonnet.src import test_utils
import tensorflow as tf


class TestCheckpoint(object):
  """Wraps a tf.train.Checkpoint to make it more convenient for testing."""

  def __init__(self, golden=None, **kwargs):
    if golden is None:
      root = absltest.get_default_test_tmpdir()
    else:
      root = os.path.join("sonnet/golden_checkpoints/",
                          golden.name)
    self._root = root
    self._prefix = os.path.join(self._root, "checkpoint")
    self._checkpoint = tf.train.Checkpoint(**kwargs)

  def save(self):
    self._checkpoint.save(file_prefix=self._prefix)

  def restore_latest(self, assert_consumed):
    status = self._checkpoint.restore(tf.train.latest_checkpoint(self._root))
    if assert_consumed:
      # Ensures that all values in the checkpoint have been consumed by some
      # checkpointable Python object.
      status.assert_consumed()
    return status


def all_goldens(test_method):
  cases = ((name, cls()) for _, name, cls in goldens.list_goldens())
  return parameterized.named_parameters(cases)(test_method)


def mirrored_all_devices(device_type):
  # NOTE: The explicit device list is required since currently MirroredStrategy
  # only considers CPU and GPU devices. This means on TPU by default we only
  # mirror on the local CPU.
  devices = tf.config.experimental.list_logical_devices(device_type=device_type)
  devices = [d.name for d in devices]
  logging.info("Mirroring over %s", devices)
  return tf.distribute.MirroredStrategy(devices=devices)


def with_soft_placement(f):
  """Wraps `f` such that it runs with soft device placement."""
  def wrapper(*a, **k):
    with tf.device(None):
      return f(*a, **k)
  return wrapper


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
    checkpoint.restore_latest(assert_consumed=True)
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
    checkpoint_2.restore_latest(assert_consumed=True)

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
      self.assertAllEqual(variable.read_value(), goldens.range_like(variable))

    # Assert the output from both modules is the same.
    # TODO(tomhennigan) Handle modules with nested outputs.
    if golden.deterministic:
      self.assertAllClose(golden.forward(module_1), golden.forward(module_2))

  @all_goldens
  def test_restore_golden(self, golden):
    """Test restoring from a golden checkpoint still works."""
    module = golden.create_module()
    checkpoint = TestCheckpoint(golden=golden, module=module)
    variables = golden.create_all_variables(module)
    for variable in variables:
      variable.assign(tf.zeros_like(variable))
    checkpoint.restore_latest(assert_consumed=True)
    for variable in variables:
      self.assertAllEqual(variable.read_value(), goldens.range_like(variable))


class DistributionStrategyCheckpointTest(test_utils.TestCase,
                                         parameterized.TestCase):

  @all_goldens
  def test_checkpoint_mirrored_strategy(self, golden):
    strategy = mirrored_all_devices(self.primary_device)
    self.assertCheckpointWithStrategy(golden, strategy, use_function=False)

  @all_goldens
  def test_checkpoint_mirrored_strategy_function(self, golden):
    strategy = mirrored_all_devices(self.primary_device)
    self.assertCheckpointWithStrategy(golden, strategy, use_function=True)

  @all_goldens
  def test_checkpoint_tpu_strategy(self, golden):
    if self.primary_device != "TPU":
      self.skipTest("Test requires a TPU")

    strategy = tf.distribute.experimental.TPUStrategy()
    self.assertCheckpointWithStrategy(golden, strategy, use_function=True)

  def assertCheckpointWithStrategy(self, golden, strategy, use_function):
    self.assertSaveRestore(golden, strategy, use_function)
    self.assertRestoreFromGolden(golden, strategy)
    self.assertRestoreFromNonDistributed(golden, strategy, use_function)
    self.assertRestoreOnCreate(golden, strategy)
    if self.primary_device != "TPU":
      # TODO(b/130555244) Enable on TPU when functions can create variables.
      self.assertRestoreOnCreateInReplicaContext(golden, strategy, use_function)

  def assertSaveRestore(self, golden, strategy, use_function):
    with strategy.scope():
      module = golden.create_module()
      variables = golden.create_all_variables(module)

    def forward():
      per_replica = strategy.experimental_run_v2(lambda: golden.forward(module))
      return tf.stack(strategy.unwrap(per_replica), axis=0)

    if use_function:
      forward = tf.function(forward)
      if self.primary_device == "TPU":
        # TODO(b/132329316) Remove when `xla.compile` allows tf.device(TPU).
        forward = with_soft_placement(forward)

    # Assign sequential values to the weights and compute a forward pass.
    for index, variable in enumerate(variables):
      variable.assign(goldens.range_like(variable, start=index))
    before_save_ys = forward()

    # Create a checkpoint and save the weights.
    checkpoint = TestCheckpoint(module=module)
    checkpoint.save()

    # Assign different values into the weights and do another forward pass. The
    # result should be different.
    for variable in variables:
      variable.assign(-tf.ones_like(variable))

    if golden.deterministic:
      y = forward()
      self.assertNotAllClose(y, before_save_ys)

    # Restore from the checkpoint and assert the module is in the same state.
    checkpoint.restore_latest(assert_consumed=True)

    for index, variable in enumerate(variables):
      # Parameters should be restored to their previous values.
      self.assertAllEqual(variable.read_value(),
                          goldens.range_like(variable, start=index))

    if golden.deterministic:
      self.assertAllEqual(forward(), before_save_ys)

  def assertRestoreFromGolden(self, golden, strategy):
    with strategy.scope():
      module = golden.create_module()
      variables = golden.create_all_variables(module)
    checkpoint = TestCheckpoint(golden=golden, module=module)
    checkpoint.restore_latest(assert_consumed=True)
    for variable in variables:
      self.assertAllEqual(variable.read_value(), goldens.range_like(variable))

  def assertRestoreFromNonDistributed(self, golden, strategy, use_function):
    # Save a checkpoint from a non-distributed model.
    module = golden.create_module()
    normal_variables = golden.create_all_variables(module)
    for index, variable in enumerate(normal_variables):
      variable.assign(goldens.range_like(variable, start=(index + 1)))
    checkpoint = TestCheckpoint(module=module)
    checkpoint.save()

    def run_forward(module):
      forward = lambda: golden.forward(module)
      if use_function:
        forward = tf.function(forward)
        if self.primary_device == "TPU":
          # TODO(b/132329316) Remove when `xla.compile` allows tf.device(TPU).
          forward = with_soft_placement(forward)
      return forward()

    if golden.deterministic:
      y_before = run_forward(module)

    # Create the same model (new params) in the strategy scope.
    with strategy.scope():
      module = golden.create_module()
      strategy_variables = golden.create_all_variables(module)

    # Ensure the distributed params are != the values in the checkpoint.
    for normal, distributed in zip(normal_variables, strategy_variables):
      distributed.assign(tf.zeros_like(distributed))
      self.assertNotAllClose(normal.read_value(), distributed.read_value())

    # Restore the checkpoint and ensure the parameters are the same.
    checkpoint = TestCheckpoint(module=module)
    checkpoint.restore_latest(assert_consumed=True)

    for normal, distributed in zip(normal_variables, strategy_variables):
      self.assertAllEqual(normal.read_value(), distributed.read_value())

    if golden.deterministic:
      y_after = run_forward(module)
      self.assertAllEqual(y_before, y_after)

  def assertRestoreOnCreate(self, golden, strategy):
    # Save a checkpoint from a non-distributed model.
    module = golden.create_module()
    normal_variables = golden.create_all_variables(module)
    for index, variable in enumerate(normal_variables):
      variable.assign(goldens.range_like(variable, start=(index + 1)))
    checkpoint = TestCheckpoint(module=module)
    checkpoint.save()

    # Create the same model (new params) in the strategy scope.
    with strategy.scope():
      module = golden.create_module()
      checkpoint = TestCheckpoint(module=module)
      status = checkpoint.restore_latest(assert_consumed=False)
      golden.forward(module)
      status.assert_consumed()
      strategy_variables = golden.create_all_variables(module)

    for normal, distributed in zip(normal_variables, strategy_variables):
      self.assertAllEqual(normal.read_value(), distributed.read_value())

  def assertRestoreOnCreateInReplicaContext(self, golden, strategy,
                                            use_function):
    with strategy.scope():
      module = golden.create_module()

    def forward():
      return strategy.experimental_run_v2(lambda: golden.forward(module))

    if use_function:
      forward = tf.function(forward)
      if self.primary_device == "TPU":
        # TODO(b/132329316) Remove when `xla.compile` allows tf.device(TPU).
        forward = with_soft_placement(forward)

    checkpoint = TestCheckpoint(golden=golden, module=module)
    status = checkpoint.restore_latest(assert_consumed=False)
    result = forward()
    status.assert_consumed()

    if golden.deterministic:
      result_iter = iter(strategy.experimental_local_results(result))
      first_replica = next(result_iter)
      for next_replica in result_iter:
        self.assertAllEqual(first_replica, next_replica)

    variables = golden.create_all_variables(module)
    for variable in variables:
      self.assertAllEqual(variable.read_value(), goldens.range_like(variable))


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
      # The output before and after saving should be close.
      self.assertAllClose(y1, y2)

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


class PublicSymbolsTest(test_utils.TestCase):

  def test_src_not_exported(self):
    self.assertFalse(hasattr(snt, "src"))


def setUpModule():
  # If a physical GPU is available make sure TF sees at least two.
  gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
  if len(gpus) == 1:
    logging.info("Splitting one physical GPU into two logical GPUs.")
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
         tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

if __name__ == "__main__":
  # tf.enable_v2_behavior()
  tf.test.main()
