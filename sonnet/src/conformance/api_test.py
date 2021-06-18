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
"""Tests for Sonnet's public API."""

import importlib

import sonnet as snt
from sonnet.src import test_utils
import tensorflow as tf


class PublicSymbolsTest(test_utils.TestCase):

  def test_src_not_exported(self):
    self.assertFalse(hasattr(snt, "src"))

  def test_supports_reload(self):
    mysnt = snt
    for _ in range(2):
      mysnt = importlib.reload(mysnt)
      self.assertFalse(hasattr(mysnt, "src"))


if __name__ == "__main__":
  tf.test.main()
