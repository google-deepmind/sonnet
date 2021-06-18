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

"""Tests Sonnet 2 with TF1."""

import sonnet as snt
from sonnet.src import test_utils
import tensorflow.compat.v1 as tf


class TensorFlow1Test(test_utils.TestCase):

  def test_requires_tf2(self):
    if tf.version.GIT_VERSION != "unknown":
      self.skipTest("This test only runs if testing against TF at head.")

    with self.assertRaisesRegex(AssertionError, "requires TensorFlow 2"):
      snt.Module()

if __name__ == "__main__":
  tf.disable_v2_behavior()
  tf.test.main()
