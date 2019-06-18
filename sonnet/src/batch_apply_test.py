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

"""Tests for sonnet.v2.src.batch_apply."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sonnet.src import base
from sonnet.src import batch_apply
from sonnet.src import test_utils
import tensorflow as tf


class BatchApplyTest(test_utils.TestCase):

  def test_simple(self):
    m = batch_apply.BatchApply(AddOne())
    x = tf.zeros([2, 3, 4])
    y = m(x)
    self.assertAllEqual(y, tf.ones([2, 3, 4]))

  def test_no_output(self):
    m = batch_apply.BatchApply(NoOutputModule())
    y = m(tf.ones([1, 1, 1]))
    self.assertIsNone(y)


class NoOutputModule(base.Module):

  def __call__(self, x):
    return None


class AddOne(base.Module):

  def __call__(self, x):
    assert len(x.shape) == 2, "Requires rank 2 input."
    return x + 1.

if __name__ == "__main__":
  # tf.enable_v2_behavior()
  tf.test.main()
