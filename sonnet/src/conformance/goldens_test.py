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
"""Tests goldens cover all modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

import sonnet as snt
from sonnet.src import test_utils
from sonnet.src.conformance import goldens
import tensorflow as tf


class CoverageTest(test_utils.TestCase):

  def test_all_modules_covered(self):
    no_checkpoint_whitelist = set([
        # TODO(petebu): Remove this once optimizer goldens check works.
        snt.optimizers.Adam,
        snt.optimizers.Momentum,
        snt.optimizers.RMSProp,
        snt.optimizers.SGD,

        # Stateless or abstract.
        snt.BatchApply,
        snt.DeepRNN,
        snt.Deferred,
        snt.Flatten,
        snt.Metric,
        snt.Module,
        snt.Optimizer,
        snt.Reshape,
        snt.RNNCore,
        snt.Sequential,
        snt.UnrolledRNN,

        # Tested via snt.nets.ResNet
        snt.nets.ResNet50,
        snt.nets.resnet.BottleNeckBlockV1,
        snt.nets.resnet.BottleNeckBlockV2,
        snt.nets.resnet.BlockGroup
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
