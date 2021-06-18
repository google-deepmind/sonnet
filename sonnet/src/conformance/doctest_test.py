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
"""Ensures that code samples in Sonnet are accurate."""

import doctest
import inspect

from absl.testing import parameterized
import sonnet as snt
from sonnet.src import test_utils
import tensorflow as tf
import tree


class DoctestTest(test_utils.TestCase, parameterized.TestCase):

  # Avoid running doctests inside a `with tf.device` block.
  ENTER_PRIMARY_DEVICE = False

  def setUp(self):
    super().setUp()
    if self.primary_device != "TPU":
      # `TpuReplicator` cannot be constructed without a TPU, however it has
      # exactly the same API as `Replicator` so we can run doctests using that
      # instead.
      snt.distribute.TpuReplicator = snt.distribute.Replicator

  @parameterized.named_parameters(test_utils.find_sonnet_python_modules(snt))
  def test_doctest(self, module):
    # `snt` et al import all dependencies from `src`, however doctest does not
    # test imported deps so we must manually set `__test__` such that imported
    # symbols are tested.
    # See: docs.python.org/3/library/doctest.html#which-docstrings-are-examined
    if not hasattr(module, "__test__") or not module.__test__:
      module.__test__ = {}
    for name in module.__all__:
      value = getattr(module, name)
      if not inspect.ismodule(value):
        if (inspect.isclass(value) or isinstance(value, str) or
            inspect.isfunction(value) or inspect.ismethod(value)):
          module.__test__[name] = value
        elif hasattr(value, "__doc__"):
          module.__test__[name] = value.__doc__

    num_failed, num_attempted = doctest.testmod(
        module,
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
        extraglobs={
            "snt": snt,
            "tf": tf,
            "tree": tree,
        })
    if num_attempted == 0:
      self.skipTest("No doctests in %s" % module.__name__)
    self.assertEqual(num_failed, 0, "{} doctests failed".format(num_failed))


if __name__ == "__main__":
  tf.test.main()
