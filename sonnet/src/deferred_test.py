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
"""Tests for sonnet.v2.src.deferred."""

from sonnet.src import base
from sonnet.src import deferred
from sonnet.src import test_utils
import tensorflow as tf


class DeferredTest(test_utils.TestCase):

  def test_target(self):
    target = ExampleModule()
    mod = deferred.Deferred(lambda: target)
    self.assertIs(mod.target, target)

  def test_only_computes_target_once(self):
    target = ExampleModule()
    targets = [target]
    mod = deferred.Deferred(targets.pop)
    for _ in range(10):
      # If target was recomputed more than once pop should fail.
      self.assertIs(mod.target, target)
      self.assertEmpty(targets)

  def test_attr_forwarding_fails_before_construction(self):
    mod = deferred.Deferred(ExampleModule)
    with self.assertRaises(AttributeError):
      getattr(mod, "foo")

  def test_getattr(self):
    mod = deferred.Deferred(ExampleModule)
    mod()
    self.assertIs(mod.w, mod.target.w)

  def test_setattr(self):
    mod = deferred.Deferred(ExampleModule)
    mod()
    new_w = tf.ones_like(mod.w)
    mod.w = new_w
    self.assertIs(mod.w, new_w)
    self.assertIs(mod.target.w, new_w)

  def test_setattr_on_target(self):
    mod = deferred.Deferred(ExampleModule)
    mod()
    w = tf.ones_like(mod.w)
    mod.w = None
    # Assigning to the target directly should reflect in the parent.
    mod.target.w = w
    self.assertIs(mod.w, w)
    self.assertIs(mod.target.w, w)

  def test_delattr(self):
    mod = deferred.Deferred(ExampleModule)
    mod()
    self.assertTrue(hasattr(mod.target, "w"))
    del mod.w
    self.assertFalse(hasattr(mod.target, "w"))

  def test_alternative_forward(self):
    mod = deferred.Deferred(AlternativeForwardModule, call_methods=("forward",))
    self.assertEqual(mod.forward(), 42)

  def test_alternative_forward_call_type_error(self):
    mod = deferred.Deferred(AlternativeForwardModule, call_methods=("forward",))
    msg = "'AlternativeForwardModule' object is not callable"
    with self.assertRaisesRegex(TypeError, msg):
      mod()

  def test_name_scope(self):
    mod = deferred.Deferred(ExampleModule)
    mod()
    self.assertEqual(mod.name_scope.name, "deferred/")
    self.assertEqual(mod.target.name_scope.name, "example_module/")

  def test_str(self):
    m = ExampleModule()
    d = deferred.Deferred(lambda: m)
    self.assertEqual("Deferred(%s)" % m, str(d))

  def test_repr(self):
    m = ExampleModule()
    d = deferred.Deferred(lambda: m)
    self.assertEqual("Deferred(%r)" % m, repr(d))


class ExampleModule(base.Module):

  def __init__(self):
    super().__init__()
    self.w = tf.Variable(1.)

  def __str__(self):
    return "ExampleModuleStr"

  def __repr__(self):
    return "ExampleModuleRepr"

  def __call__(self):
    return self.w


class AlternativeForwardModule(base.Module):

  def forward(self):
    return 42


if __name__ == "__main__":
  tf.test.main()
