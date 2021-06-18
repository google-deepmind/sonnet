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


import doctest

from sonnet.src import base
from sonnet.src import custom_getter
from sonnet.src import test_utils
import tensorflow as tf


class CustomVariableGetterTest(test_utils.TestCase):

  def testDoesNotModifyNonVariables(self):

    class MyModule(base.Module):
      v = tf.Variable(21.)
      d = {}

    my_module = MyModule()
    self.assertEqual(21., self.evaluate(my_module.v))

    with custom_getter.custom_variable_getter(lambda v: v * 2):
      self.assertEqual(42., self.evaluate(my_module.v))
      my_module.d["foo"] = "bar"

    self.assertEqual(21., self.evaluate(my_module.v))
    self.assertEqual("bar", my_module.d["foo"])


class DoctestTest(test_utils.TestCase):

  def testDoctest(self):
    num_failed, num_attempted = doctest.testmod(
        custom_getter, extraglobs={"snt": base})
    self.assertGreater(num_attempted, 0, "No doctests found.")
    self.assertEqual(num_failed, 0, "{} doctests failed".format(num_failed))


if __name__ == "__main__":
  tf.test.main()
