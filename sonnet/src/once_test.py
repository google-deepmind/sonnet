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
"""Tests for sonnet.v2.src.once."""

import pickle

from absl.testing import absltest
from absl.testing import parameterized
from sonnet.src import once


class OnceTest(parameterized.TestCase):

  def test_runs_once(self):
    r = []

    @once.once
    def f():
      r.append(None)

    for _ in range(3):
      f()

    self.assertEqual(r, [None])

  def test_always_returns_none(self):
    f = once.once(lambda: "Hello, world!")
    with self.assertRaisesRegex(ValueError, "snt.once .* cannot return"):
      f()

  def test_does_not_cache_on_error(self):

    @once.once
    def f():
      raise ValueError

    with self.assertRaises(ValueError):
      f()
    with self.assertRaises(ValueError):
      f()

  def test_method(self):
    o1 = Counter()
    o2 = Counter()
    for _ in range(10):
      o1.increment()
      o2.increment()

    self.assertEqual(o1.call_count, 1)
    self.assertEqual(o2.call_count, 1)

  def test_method_does_not_cache_on_error(self):

    class Dummy:

      @once.once
      def f(self):
        raise ValueError

    o = Dummy()
    with self.assertRaises(ValueError):
      o.f()
    with self.assertRaises(ValueError):
      o.f()

  def test_pickle_method_before_evaluation(self):
    c1 = Counter()
    c2 = pickle.loads(pickle.dumps(c1))
    c1.increment()
    self.assertEqual(c1.call_count, 1)
    self.assertEqual(c2.call_count, 0)
    c2.increment()
    self.assertEqual(c1.call_count, 1)
    self.assertEqual(c2.call_count, 1)

  def test_pickle_method_already_evaluated(self):
    c1 = Counter()
    c1.increment()
    self.assertEqual(c1.call_count, 1)
    c2 = pickle.loads(pickle.dumps(c1))
    self.assertEqual(c2.call_count, 1)
    c2.increment()
    self.assertEqual(c2.call_count, 1)

  def test_inline(self):
    r = []
    f = once.once(lambda: r.append(None))
    for _ in range(10):
      f()
    self.assertEqual(r, [None])

  @parameterized.named_parameters(
      ("lambda", lambda: lambda: None), ("function", lambda: nop),
      ("method", lambda: NoOpCallable().nop),
      ("special_method", lambda: NoOpCallable().__call__),
      ("object", lambda: NoOpCallable()))  # pylint: disable=unnecessary-lambda
  def test_adds_property(self, factory):
    f = factory()
    self.assertIs(once.once(f).__snt_once_wrapped__, f)


def nop():
  pass


class NoOpCallable:

  def nop(self):
    pass

  def __call__(self):
    pass


class Counter:
  call_count = 0

  @once.once
  def increment(self):
    self.call_count += 1


if __name__ == "__main__":
  absltest.main()
