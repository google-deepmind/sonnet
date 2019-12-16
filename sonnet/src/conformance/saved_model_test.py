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
"""Tests using tf.saved_model and Sonnet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import absltest
from absl.testing import parameterized
import sonnet as snt
from sonnet.src import test_utils
from sonnet.src.conformance import goldens
import tensorflow as tf
import tree


class SavedModelTest(test_utils.TestCase, parameterized.TestCase):

  @goldens.all_goldens
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
    saved_model.inference(x)
    v1 = saved_model.all_variables

    # Save the model to disk and restore it.
    tmp_dir = os.path.join(absltest.get_default_test_tmpdir(), golden.name)
    tf.saved_model.save(saved_model, tmp_dir)
    restored_model = tf.saved_model.load(tmp_dir)

    # Run the loaded model and pull variable values.
    v2 = restored_model.all_variables
    y2 = restored_model.inference(x)

    if golden.deterministic:
      # The output from both the saved and restored model should be close.
      y1 = saved_model.inference(x)
      tree.map_structure(self.assertAllEqual, y1, y2)

    for a, b in zip(v1, v2):
      self.assertEqual(a.name, b.name)
      self.assertEqual(a.device, b.device)
      self.assertAllEqual(a.read_value(), b.read_value())


if __name__ == "__main__":
  # tf.enable_v2_behavior()
  tf.test.main()
