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

"""Doctests for Sonnet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import doctest

from absl.testing import parameterized
import sonnet as snt
from sonnet.src import test_utils
import tensorflow as tf


class DoctestTest(test_utils.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(test_utils.find_sonnet_python_modules(snt))
  def test_doctest(self, module):
    num_failed, num_attempted = doctest.testmod(
        module, optionflags=doctest.ELLIPSIS, extraglobs={"snt": snt, "tf": tf})
    if num_attempted == 0:
      self.skipTest("No doctests in %s" % module.__name__)
    self.assertEqual(num_failed, 0, "{} doctests failed".format(num_failed))

if __name__ == "__main__":
  # tf.enable_v2_behavior()
  tf.test.main()
