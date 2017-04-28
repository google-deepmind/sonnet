# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for sonnet.testing.parameterized."""

import collections
import unittest

# Dependency imports
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from sonnet.testing import parameterized

from tensorflow.python.platform import googletest


class MyOwnClass(object):
  pass


def DictDecorator(key, value):
  """Sample implementation of a chained decorator.

  Sets a single field in a dict on a test with a dict parameter.
  Uses the exposed '_ParameterizedTestIter.testcases' field to
  modify arguments from previous decorators to allow decorator chains.

  Args:
    key: key to map to
    value: value to set

  Returns:
    The test decorator
  """
  def Decorator(test_method):
    # If decorating result of another DictDecorator
    if isinstance(test_method, collections.Iterable):
      actual_tests = []
      for old_test in test_method.testcases:
        # each test is a ('test_suffix', dict) tuple
        new_dict = old_test[1].copy()
        new_dict[key] = value
        test_suffix = '%s_%s_%s' % (old_test[0], key, value)
        actual_tests.append((test_suffix, new_dict))

      test_method.testcases = actual_tests
      return test_method
    else:
      test_suffix = ('_%s_%s') % (key, value)
      tests_to_make = ((test_suffix, {key: value}),)
      # 'test_method' here is the original test method
      return parameterized.NamedParameters(*tests_to_make)(test_method)
  return Decorator


class ParameterizedTestsTest(googletest.TestCase):
  # The test testcases are nested so they're not
  # picked up by the normal test case loader code.

  class GoodAdditionParams(parameterized.ParameterizedTestCase):

    @parameterized.Parameters(
        (1, 2, 3),
        (4, 5, 9))
    def testAddition(self, op1, op2, result):
      self.arguments = (op1, op2, result)
      self.assertEqual(result, op1 + op2)

  # This class does not inherit from ParameterizedTestCase.
  class BadAdditionParams(googletest.TestCase):

    @parameterized.Parameters(
        (1, 2, 3),
        (4, 5, 9))
    def testAddition(self, op1, op2, result):
      pass  # Always passes, but not called w/out ParameterizedTestCase.

  class MixedAdditionParams(parameterized.ParameterizedTestCase):

    @parameterized.Parameters(
        (1, 2, 1),
        (4, 5, 9))
    def testAddition(self, op1, op2, result):
      self.arguments = (op1, op2, result)
      self.assertEqual(result, op1 + op2)

  class DictionaryArguments(parameterized.ParameterizedTestCase):

    @parameterized.Parameters(
        {'op1': 1, 'op2': 2, 'result': 3},
        {'op1': 4, 'op2': 5, 'result': 9})
    def testAddition(self, op1, op2, result):
      self.assertEqual(result, op1 + op2)

  class NoParameterizedTests(parameterized.ParameterizedTestCase):
    # iterable member with non-matching name
    a = 'BCD'
    # member with matching name, but not a generator
    testInstanceMember = None  # pylint: disable=invalid-name

    # member with a matching name and iterator, but not a generator
    testString = 'foo'  # pylint: disable=invalid-name

    # generator, but no matching name
    def someGenerator(self):  # pylint: disable=invalid-name
      yield
      yield
      yield

    # Generator function, but not a generator instance.
    def testGenerator(self):
      yield
      yield
      yield

    def testNormal(self):
      self.assertEqual(3, 1 + 2)

  class GeneratorTests(parameterized.ParameterizedTestCase):

    def generateTestCases():  # pylint: disable=no-method-argument,invalid-name
      for _ in xrange(10):
        yield lambda x: None

    testGeneratedTestCases = generateTestCases()  # pylint: disable=invalid-name

  class ArgumentsWithAddresses(parameterized.ParameterizedTestCase):

    @parameterized.Parameters(
        (object(),),
        (MyOwnClass(),),
    )
    def testSomething(self, unused_obj):
      pass

  class NamedTests(parameterized.ParameterizedTestCase):

    @parameterized.NamedParameters(
        ('Interesting', 0),
        ('Boring', 1),
    )
    def testSomething(self, unused_obj):
      pass

    def testWithoutParameters(self):
      pass

  class UnderscoreNamedTests(parameterized.ParameterizedTestCase):
    """Example tests using PEP-8 style names instead of camel-case."""

    @parameterized.NamedParameters(
        ('interesting', 0),
        ('boring', 1),
    )
    def test_something(self, unused_obj):
      pass

    def test_without_parameters(self):
      pass

  class ChainedTests(parameterized.ParameterizedTestCase):

    @DictDecorator('cone', 'waffle')
    @DictDecorator('flavor', 'strawberry')
    def testChained(self, dictionary):
      self.assertDictEqual(dictionary, {'cone': 'waffle',
                                        'flavor': 'strawberry'})

  class SingletonListExtraction(parameterized.ParameterizedTestCase):

    @parameterized.Parameters(
        (i, i * 2) for i in range(10))
    def testSomething(self, unused_1, unused_2):
      pass

  class SingletonArgumentExtraction(parameterized.ParameterizedTestCase):

    @parameterized.Parameters(1, 2, 3, 4, 5, 6)
    def testNumbers(self, unused_1):
      pass

    @parameterized.Parameters('foo', 'bar', 'baz')
    def testStrings(self, unused_1):
      pass

  @parameterized.Parameters(
      (1, 2, 3),
      (4, 5, 9))
  class DecoratedClass(parameterized.ParameterizedTestCase):

    def testAdd(self, arg1, arg2, arg3):
      self.assertEqual(arg1 + arg2, arg3)

    def testSubtractFail(self, arg1, arg2, arg3):
      self.assertEqual(arg3 + arg2, arg1)

  @parameterized.Parameters(
      (a, b, a+b) for a in range(1, 5) for b in range(1, 5))
  class GeneratorDecoratedClass(parameterized.ParameterizedTestCase):

    def testAdd(self, arg1, arg2, arg3):
      self.assertEqual(arg1 + arg2, arg3)

    def testSubtractFail(self, arg1, arg2, arg3):
      self.assertEqual(arg3 + arg2, arg1)

  @parameterized.Parameters(
      (1, 2, 3),
      (4, 5, 9),
  )
  class DecoratedBareClass(googletest.TestCase):

    def testAdd(self, arg1, arg2, arg3):
      self.assertEqual(arg1 + arg2, arg3)

  class OtherDecorator(parameterized.ParameterizedTestCase):

    @unittest.skip('wraps _ParameterizedTestIter')
    @parameterized.Parameters((1), (2))
    def testOtherThenParameterized(self, arg1):
      pass

    @parameterized.Parameters((1), (2))
    @unittest.skip('is wrapped by _ParameterizedTestIter')
    def testParameterizedThenOther(self, arg1):
      pass

  def testMissingInheritance(self):
    ts = unittest.makeSuite(self.BadAdditionParams)
    self.assertEqual(1, ts.countTestCases())

    res = unittest.TestResult()
    ts.run(res)
    self.assertEqual(1, res.testsRun)
    self.assertFalse(res.wasSuccessful())
    self.assertIn('without having inherited', str(res.errors[0]))

  def testCorrectExtractionNumbers(self):
    ts = unittest.makeSuite(self.GoodAdditionParams)
    self.assertEqual(2, ts.countTestCases())

  def testSuccessfulExecution(self):
    ts = unittest.makeSuite(self.GoodAdditionParams)
    res = unittest.TestResult()
    ts.run(res)
    self.assertEqual(2, res.testsRun)
    self.assertTrue(res.wasSuccessful())

  def testCorrectArguments(self):
    ts = unittest.makeSuite(self.GoodAdditionParams)
    res = unittest.TestResult()

    params = set([
        (1, 2, 3),
        (4, 5, 9)])
    for test in ts:
      test(res)
      self.assertIn(test.arguments, params)
      params.remove(test.arguments)
    self.assertEqual(0, len(params))

  def testRecordedFailures(self):
    ts = unittest.makeSuite(self.MixedAdditionParams)
    self.assertEqual(2, ts.countTestCases())

    res = unittest.TestResult()
    ts.run(res)
    self.assertEqual(2, res.testsRun)
    self.assertFalse(res.wasSuccessful())
    self.assertEqual(1, len(res.failures))
    self.assertEqual(0, len(res.errors))

  def testId(self):
    ts = unittest.makeSuite(self.ArgumentsWithAddresses)
    self.assertEqual(
        '__main__.ArgumentsWithAddresses.testSomething(<object>)',
        list(ts)[0].id())
    ts = unittest.makeSuite(self.GoodAdditionParams)
    self.assertEqual(
        '__main__.GoodAdditionParams.testAddition(1, 2, 3)',
        list(ts)[0].id())

  def testDictParameters(self):
    ts = unittest.makeSuite(self.DictionaryArguments)
    res = unittest.TestResult()
    ts.run(res)
    self.assertEqual(2, res.testsRun)
    self.assertTrue(res.wasSuccessful())

  def testGeneratorTests(self):
    ts = unittest.makeSuite(self.GeneratorTests)
    self.assertEqual(10, ts.countTestCases())

  def testNamedParametersRun(self):
    ts = unittest.makeSuite(self.NamedTests)
    self.assertEqual(3, ts.countTestCases())
    res = unittest.TestResult()
    ts.run(res)
    self.assertEqual(3, res.testsRun)
    self.assertTrue(res.wasSuccessful())

  def testNamedParametersId(self):
    ts = sorted(unittest.makeSuite(self.NamedTests),
                key=lambda t: t.id())
    self.assertEqual(
        '__main__.NamedTests.testSomethingBoring',
        ts[0].id())
    self.assertEqual(
        '__main__.NamedTests.testSomethingInteresting',
        ts[1].id())

  def testNamedParametersIdWithUnderscoreCase(self):
    ts = sorted(unittest.makeSuite(self.UnderscoreNamedTests),
                key=lambda t: t.id())
    self.assertEqual(
        '__main__.UnderscoreNamedTests.test_something_boring',
        ts[0].id())
    self.assertEqual(
        '__main__.UnderscoreNamedTests.test_something_interesting',
        ts[1].id())

  def testLoadNamedTest(self):
    loader = unittest.TestLoader()
    ts = list(loader.loadTestsFromName('NamedTests.testSomethingInteresting',
                                       module=self))
    self.assertEqual(1, len(ts))
    self.assertTrue(ts[0].id().endswith('.testSomethingInteresting'))

  def testDuplicateNamedTestFails(self):
    with self.assertRaises(AssertionError):

      class _(parameterized.ParameterizedTestCase):

        @parameterized.NamedParameters(
            ('Interesting', 0),
            ('Interesting', 1),
        )
        def testSomething(self, unused_obj):
          pass

  def testParameterizedTestIterHasTestcasesProperty(self):
    @parameterized.Parameters(1, 2, 3, 4, 5, 6)
    def testSomething(unused_self, unused_obj):  # pylint: disable=invalid-name
      pass

    expected_testcases = [1, 2, 3, 4, 5, 6]
    self.assertTrue(hasattr(testSomething, 'testcases'))
    assert_items_equal = (self.assertCountEqual if six.PY3
                          else self.assertItemsEqual)
    assert_items_equal(expected_testcases, testSomething.testcases)

  def testChainedDecorator(self):
    ts = unittest.makeSuite(self.ChainedTests)
    self.assertEqual(1, ts.countTestCases())
    test = next(t for t in ts)
    self.assertTrue(hasattr(test, 'testChained_flavor_strawberry_cone_waffle'))
    res = unittest.TestResult()

    ts.run(res)
    self.assertEqual(1, res.testsRun)
    self.assertTrue(res.wasSuccessful())

  def testSingletonListExtraction(self):
    ts = unittest.makeSuite(self.SingletonListExtraction)
    res = unittest.TestResult()
    ts.run(res)
    self.assertEqual(10, res.testsRun)
    self.assertTrue(res.wasSuccessful())

  def testSingletonArgumentExtraction(self):
    ts = unittest.makeSuite(self.SingletonArgumentExtraction)
    res = unittest.TestResult()
    ts.run(res)
    self.assertEqual(9, res.testsRun)
    self.assertTrue(res.wasSuccessful())

  def testDecoratedBareClass(self):
    ts = unittest.makeSuite(self.DecoratedBareClass)
    res = unittest.TestResult()
    ts.run(res)
    self.assertEqual(2, res.testsRun)
    self.assertTrue(res.wasSuccessful(), msg=str(res.failures))

  def testDecoratedClass(self):
    ts = unittest.makeSuite(self.DecoratedClass)
    res = unittest.TestResult()
    ts.run(res)
    self.assertEqual(4, res.testsRun)
    self.assertEqual(2, len(res.failures))

  def testGeneratorDecoratedClass(self):
    ts = unittest.makeSuite(self.GeneratorDecoratedClass)
    res = unittest.TestResult()
    ts.run(res)
    self.assertEqual(32, res.testsRun)
    self.assertEqual(16, len(res.failures))

  def testNoDuplicateDecorations(self):
    with self.assertRaises(AssertionError):

      @parameterized.Parameters(1, 2, 3, 4)
      class _(parameterized.ParameterizedTestCase):

        @parameterized.Parameters(5, 6, 7, 8)
        def testSomething(self, unused_obj):
          pass

  def testOtherDecoratorOrdering(self):
    ts = unittest.makeSuite(self.OtherDecorator)
    res = unittest.TestResult()
    ts.run(res)
    # Two for when the parameterized tests call the skip wrapper.
    # One for when the skip wrapper is called first and doesn't iterate.
    self.assertEqual(3, res.testsRun)
    self.assertTrue(res.wasSuccessful(), msg=str(res.failures))


def _DecorateWithSideEffects(func, self):
  self.sideeffect = True
  func(self)


if __name__ == '__main__':
  unittest.main()
