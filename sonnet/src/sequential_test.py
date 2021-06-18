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
"""Tests for sonnet.v2.src.sequential."""

from absl.testing import parameterized
from sonnet.src import sequential
from sonnet.src import test_utils
import tensorflow as tf

input_parameters = parameterized.parameters(object(), ([[[1.]]],), ({1, 2, 3},),
                                            None, "str", 1)


class SequentialTest(test_utils.TestCase, parameterized.TestCase):

  @input_parameters
  def test_empty(self, value):
    net = sequential.Sequential()
    self.assertIs(net(value), value)

  @input_parameters
  def test_empty_drops_varargs_varkwargs(self, value):
    net = sequential.Sequential()
    self.assertIs(net(value, object(), keyword=object()), value)

  @input_parameters
  def test_identity_chain(self, value):
    net = sequential.Sequential([identity, identity, identity])
    self.assertIs(net(value), value)

  def test_call(self):
    seq = sequential.Sequential([append_character(ch) for ch in "rocks!"])
    self.assertEqual(seq("Sonnet "), "Sonnet rocks!")

  def test_varargs_varkwargs_to_call(self):
    layer1 = lambda a, b, c: ((a + b + c), (c + b + a))
    layer2 = lambda a: a[0] + "," + a[1]
    net = sequential.Sequential([layer1, layer2])
    self.assertEqual(net("a", "b", c="c"), "abc,cba")


def identity(v):
  return v


def append_character(c):
  return lambda v: v + c


if __name__ == "__main__":
  tf.test.main()
