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
"""Tests Sonnet and XLA."""

import functools

from absl.testing import parameterized
from sonnet.src import test_utils
from sonnet.src.conformance import goldens
import tensorflow as tf
import tree


class XLATest(test_utils.TestCase, parameterized.TestCase):

  @goldens.all_goldens
  def test_compile(self, golden):
    mod = golden.create_module()
    golden.create_all_variables(mod)

    @tf.function
    def forward():
      f = lambda: golden.forward(mod)
      out = tf.xla.experimental.compile(f)
      if len(out) == 1:
        return out[0]
      else:
        return out if out else None

    if self.primary_device == "TPU":
      # TODO(b/132329316) Remove when `xla.compile` allows tf.device(TPU).
      with tf.device(None):
        xla_out = forward()
      atol = golden.tpu_atol
    else:
      xla_out = forward()
      atol = 1e-3

    if golden.deterministic and not golden.has_side_effects:
      out = golden.forward(mod)
      tree.map_structure(
          functools.partial(self.assertAllClose, atol=atol), out, xla_out)

  @goldens.all_goldens
  def test_jit_scope(self, golden):
    mod = golden.create_module()
    golden.create_all_variables(mod)

    @tf.function
    def forward():
      with tf.xla.experimental.jit_scope():
        return golden.forward(mod)

    xla_out = forward()
    if self.primary_device == "TPU":
      atol = golden.tpu_atol
    else:
      atol = 1e-3

    if golden.deterministic and not golden.has_side_effects:
      out = golden.forward(mod)
      tree.map_structure(
          functools.partial(self.assertAllClose, atol=atol), out, xla_out)


if __name__ == "__main__":
  tf.test.main()
