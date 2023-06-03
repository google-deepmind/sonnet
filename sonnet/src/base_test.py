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
"""Tests for sonnet.v2.src.base."""

import abc

from absl.testing import parameterized
import numpy as np
from sonnet.src import base
from sonnet.src import test_utils
import tensorflow as tf
import wrapt


class BaseTest(test_utils.TestCase):

  def test_basic(self):
    m = LambdaModule()
    self.assertIsNone(m(None))

  def testWrappedMethod(self):
    mod = WraptModule()
    scope_name, y = mod(3)
    self.assertEqual(scope_name, "wrapt_module/")
    self.assertEqual(y, (3**2)**2)

  def testControlFlow(self):
    mod = ControlFlowModule()
    f = tf.function(mod).get_concrete_function(tf.TensorSpec([]))
    self.assertEqual(f(tf.constant(1.)).numpy(), 1.)
    self.assertEqual(f(tf.constant(11.)).numpy(), 11.**2)


class TestModuleNaming(tf.test.TestCase):

  def test_single_name(self):
    mod = base.Module(name="simple")
    self.assertEqual(mod.name, "simple")
    self.assertEqual(mod.name_scope.name, "simple/")

  def test_construct_in_scope(self):
    with tf.name_scope("foo"):
      mod = base.Module(name="bar")
    self.assertEqual(mod.name, "bar")
    self.assertEqual(mod.name_scope.name, "foo/bar/")

  def test_enters_name_scope_in_call(self):
    mod = ReturnsNameScopeModule()
    for _ in range(3):
      self.assertEqual(mod(), mod.name_scope.name)

  def test_enters_name_scope_in_other_method(self):
    mod = ReturnsNameScopeModule()
    for _ in range(3):
      self.assertEqual(mod.alternative_forward(), mod.name_scope.name)

  def test_subclassed_module(self):
    mod = SubclassedReturnsNameScopeModule()
    for _ in range(3):
      self.assertEqual(mod.alternative_forward(), mod.name_scope.name)
      self.assertEqual(mod.alternative_alternative_forward(),
                       mod.name_scope.name)

  def test_submodule_created_late(self):
    m = TreeModule()
    self.assertEqual(m.name, "tree_module")
    self.assertEqual(m.name_scope.name, "tree_module/")
    leaf1 = m.new_leaf()
    self.assertEqual(leaf1.name, "tree_module")
    self.assertEqual(leaf1.name_scope.name, "tree_module/tree_module/")

  def test_does_not_evaluate_property_methods(self):
    mod = PropertyThrowsWhenCalledModule()
    with self.assertRaises(AssertionError):
      mod.raise_assertion_error  # pylint: disable=pointless-statement

  def test_overridden_name_scope(self):
    mod = ModuleOverridingNameScope()
    self.assertEqual(mod(), mod.name_scope.name)
    self.assertEqual(mod.alternative_forward(), mod.name_scope.name)

  def test_patched_callable(self):
    with tf.name_scope("foo"):
      mod = base.Module(name="bar")
    mod.foo = get_name_scope
    # `foo` is not a method so we do not re-enter the name scope.
    self.assertEqual(mod.foo(), "")

  def test_property(self):
    mod = PropertyModule()
    mod.some_property = None, None  # None, None for the linter.
    getter_scope_name, setter_scope_name = mod.some_property
    self.assertEqual(getter_scope_name, "property_module/")
    self.assertEqual(setter_scope_name, "property_module/")

  def test_property_no_name_scope(self):
    mod = PropertyModule()
    mod.no_name_scope_property = None, None  # None, None for the linter.
    getter_scope_name, setter_scope_name = mod.no_name_scope_property
    self.assertEqual(getter_scope_name, "")
    self.assertEqual(setter_scope_name, "")

  def test_ctor_no_name_scope(self):
    mod = CtorNoNameScope()
    self.assertEqual(mod.ctor_name_scope, "")
    self.assertEqual(mod.w.name, "w:0")

  def test_ctor_no_name_scope_no_super(self):
    msg = ("Constructing a snt.Module without calling the super constructor is "
           "not supported")
    with self.assertRaisesRegex(ValueError, msg):
      CtorNoNameScopeNoSuper()

  def test_invalid_name(self):
    msg = ".* is not a valid module name"
    with self.assertRaisesRegex(ValueError, msg):
      base.Module(name="$Foo")

  def test_modules_not_numbered_in_eager(self):
    mod = RecursiveModule(2)
    self.assertEqual(mod.name_scope.name, "badger/")
    self.assertEqual(mod.child.name_scope.name, "badger/badger/")

    mod = RecursiveModule(2)
    self.assertEqual(mod.name_scope.name, "badger/")
    self.assertEqual(mod.child.name_scope.name, "badger/badger/")

  def test_module_numbering_in_graph(self):
    with tf.Graph().as_default():
      mod = RecursiveModule(2)
      self.assertEqual(mod.name_scope.name, "badger/")
      self.assertEqual(mod.child.name_scope.name, "badger/badger/")

      mod = RecursiveModule(2)
      self.assertEqual(mod.name_scope.name, "badger_1/")
      self.assertEqual(mod.child.name_scope.name, "badger_1/badger/")

  def test_ctor_error_closes_name_scope(self):
    with self.assertRaises(ErrorModuleError):
      # If super constructor is called then a name scope is opened then an error
      # is thrown. The metaclass should handle this and close the namescope
      # before re-throwing the exception.
      ErrorModule(call_super=True)

    self.assertEqual("", get_name_scope())

  def test_ctor_error_handles_ctor_not_opening_name_scope(self):
    with self.assertRaises(ErrorModuleError):
      # If super ctor is not called then the name scope isn't opened. We need to
      # ensure that this doesn't trigger an exception (e.g. the metaclass trying
      # to __exit__ a non-existent name scope).
      ErrorModule(call_super=False)

    self.assertEqual("", get_name_scope())

  def test_forward_method_closes_name_scope(self):
    mod = ErrorModule(call_super=True, raise_in_constructor=False)
    with self.assertRaises(ErrorModuleError):
      mod()

    self.assertEqual("", get_name_scope())

  def test_get_attr_doesnt_enter_name_scope(self):
    scope_names = []

    class GetAttrModule(base.Module):

      def __getattr__(self, name):
        scope_names.append((name, get_name_scope()))
        return super().__getattr__(name)

    mod = GetAttrModule()
    with self.assertRaises(AttributeError):
      mod.does_not_exist  # pylint: disable=pointless-statement
    self.assertIn(("does_not_exist", ""), scope_names)

  def test_get_attribute_doesnt_enter_name_scope(self):
    scope_names = []

    class GetAttributeModule(base.Module):

      def __getattribute__(self, name):
        scope_names.append((name, get_name_scope()))
        return super().__getattribute__(name)

    mod = GetAttributeModule()
    with self.assertRaises(AttributeError):
      mod.does_not_exist  # pylint: disable=pointless-statement
    self.assertIn(("does_not_exist", ""), scope_names)


class VariableNamingTest(tf.test.TestCase):

  def test_variable_names(self):
    mod = RecursiveModule(3)
    self.assertEqual(mod.w.name, "badger/mushroom:0")
    self.assertEqual(mod.child.w.name, "badger/badger/mushroom:0")
    self.assertEqual(mod.child.child.w.name, "badger/badger/badger/mushroom:0")


class AutoReprTest(tf.test.TestCase):

  def test_order_matches_argspec(self):
    module = RecursiveModule(trainable=False, depth=2)
    self.assertEqual(repr(module), "RecursiveModule(depth=2, trainable=False)")

  def test_defaults_ignored(self):
    module = RecursiveModule(1)
    self.assertEqual(repr(module), "RecursiveModule(depth=1)")

  def test_does_not_fail_with_hostile_input(self):
    r = RaisesOnEquality()
    self.assertFalse(r.equality_checked)
    module = NoopModule(r)
    self.assertEqual(repr(module), "NoopModule(a=hostile)")
    self.assertTrue(r.equality_checked)

  def test_args_are_repred(self):
    module = TreeModule(name="TreeModule")
    self.assertEqual(repr(module), "TreeModule(name='TreeModule')")
    module = TreeModule("TreeModule")
    self.assertEqual(repr(module), "TreeModule(name='TreeModule')")

  def test_long_repr_multi_line(self):
    module = TakesSubmodules([TreeModule() for _ in range(6)], name="hai")
    self.assertEqual(
        repr(module), "\n".join([
            "TakesSubmodules(",
            "    submodules=[TreeModule(),",
            "                TreeModule(),",
            "                TreeModule(),",
            "                TreeModule(),",
            "                TreeModule(),",
            "                TreeModule()],",
            "    name='hai',",
            ")",
        ]))

  def test_repr_wildcard(self):
    module = WildcardInit(1, 2, 3, foo="bar")
    # NOTE: This is not a valid piece of Python, but it is unambiguous and
    # probably the most helpful thing we can do. An alternative would be to
    # special case `__init__(a, *args)` and not render names preceding *args
    # but this is unlikely to be common in the ctor.
    self.assertEqual(repr(module), "WildcardInit(a=1, b=2, 3, foo='bar')")

  def test_repr_non_bool_equality(self):
    class FooModule(base.Module):

      def __init__(self, a=((-1., -1.))):
        super().__init__()

    # auto_repr tests default values for equality. In numpy (and TF2) equality
    # is tested elementwise so the return value of `==` is an ndarray which we
    # then attempt to reduce to a boolean.
    foo = FooModule(a=np.array([[2., 2.]]))
    self.assertEqual(repr(foo), "FooModule(a=array([[2., 2.]]))")
    foo = FooModule(a=np.array([[-1., -1.]]))
    self.assertEqual(repr(foo), "FooModule(a=array([[-1., -1.]]))")


class ForwardMethodsTest(tf.test.TestCase):

  def testFunctionType(self):
    mod = ModuleWithFunctionAnnotatedCall()
    self.assertIsInstance(mod.forward, base.TFFunctionType)
    self.assertIsInstance(mod.forward_ag, base.TFFunctionType)

  def testEntersNameScope_call(self):
    mod = ModuleWithFunctionAnnotatedCall()
    self.assertEqual(mod.forward().numpy(),
                     b"module_with_function_annotated_call/")
    # TODO(b/122265385) Re-enable this assertion.
    # self.assertEqual(mod.forward_ag().numpy(),
    #                  b"module_with_function_annotated_call/")

  def testEntersNameScope_concreteFunction(self):
    mod = ModuleWithFunctionAnnotatedCall()
    self.assertEqual(mod.forward.get_concrete_function()().numpy(),
                     b"module_with_function_annotated_call/")
    # TODO(b/122265385) Re-enable this assertion.
    # self.assertEqual(mod.forward_ag.get_concrete_function()().numpy(),
    #                  b"module_with_function_annotated_call/")


class AbcTest(tf.test.TestCase):

  def testAbstract(self):
    msg = "Can't instantiate .* abstract method"
    with self.assertRaisesRegex(TypeError, msg):
      AbstractModule()  # pylint: disable=abstract-class-instantiated

  def testConcrete(self):
    mod = ConcreteModule()
    x, scope_name = mod(2.)
    self.assertEqual(x, 4.)
    self.assertEqual(scope_name, "concrete_module/")
    self.assertEqual(get_name_scope(), "")

  def testCallMethodsOnParent(self):
    mod = ConcreteModule()
    self.assertEqual(mod.foo(), True)


class CustomGradientTest(test_utils.TestCase):

  def test_custom_gradient(self):
    if tf.version.GIT_VERSION != "unknown":
      # TODO(tomhennigan) Enable this once TF 2.0.1 comes out.
      self.skipTest("Requires TF > 2.0.0")

    mod = ZeroGradModule()
    with tf.GradientTape() as tape:
      y = mod(2.)
    g = tape.gradient(y, mod.w)
    self.assertAllEqual(g, tf.zeros([2, 2]))


class ZeroGradModule(base.Module):

  @tf.custom_gradient
  def __call__(self, x):
    if not hasattr(self, "w"):
      self.w = tf.Variable(tf.ones([2, 2]), name="w")

    with tf.GradientTape() as tape:
      y = tf.reduce_sum(self.w ** x)
    dw = tape.gradient(y, self.w)

    def grad(dy, variables=None):
      assert variables
      return dy * 0, [dw * 0]

    return y, grad


class LambdaModule(base.Module):

  def __call__(self, x):
    return x


def get_name_scope():
  with tf.name_scope("x") as scope_name:
    return scope_name[:-2]


@wrapt.decorator
def wrapt_decorator(method, instance, args, kwargs):
  if instance is None:
    raise ValueError("Expected instance to be non-null.")

  scope_name, y = method(*args, **kwargs)
  return scope_name, y**2


class WraptModule(base.Module):

  @wrapt_decorator
  def __call__(self, x):
    return get_name_scope(), x**2


class ControlFlowModule(base.Module):

  def __call__(self, x):
    if x < 10:
      return x
    else:
      return x**2


class ErrorModuleError(Exception):
  pass


class ErrorModule(base.Module):

  def __init__(self, call_super, raise_in_constructor=True):
    if call_super:
      super().__init__()
    if raise_in_constructor:
      raise ErrorModuleError("Deliberate error!")

  def __call__(self):
    raise ErrorModuleError("Deliberate error!")


class RecursiveModule(base.Module):

  def __init__(self, depth, trainable=True):
    super().__init__(name="badger")
    self.child = None
    if depth > 1:
      self.child = RecursiveModule(depth - 1, trainable=trainable)
    self.w = tf.Variable(1.0, trainable=trainable, name="mushroom")


class AbstractModule(base.Module, metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def __call__(self, x):
    pass

  def foo(self):
    return True


class ConcreteModule(AbstractModule):

  def __call__(self, x):
    return x**2, get_name_scope()


class TreeModule(base.Module):

  def __init__(self, name=None):
    super().__init__(name=name)
    self._leaves = []

  def new_leaf(self, name=None):
    leaf = TreeModule(name=name)
    self._leaves.append(leaf)
    return leaf


class ReturnsNameScopeModule(base.Module):

  def alternative_forward(self):
    return get_name_scope()

  def __call__(self):
    return get_name_scope()


class SubclassedReturnsNameScopeModule(ReturnsNameScopeModule):

  def alternative_alternative_forward(self):
    return get_name_scope()


class PropertyThrowsWhenCalledModule(base.Module):

  @property
  def raise_assertion_error(self):
    raise AssertionError


class ModuleOverridingNameScope(ReturnsNameScopeModule):

  @property
  def name_scope(self):
    return tf.name_scope("yolo/")


class CommonErrorsTest(test_utils.TestCase, parameterized.TestCase):

  def test_not_calling_super_constructor(self):
    msg = ("Constructing a snt.Module without calling the super constructor is "
           "not supported")
    with self.assertRaisesRegex(ValueError, msg):
      DoesNotCallSuperConstructorModule()

  def test_calls_method_before_super(self):
    msg = "super constructor must be called before any other methods"
    with self.assertRaisesRegex(AttributeError, msg):
      CallsMethodBeforeSuperConstructorModule(allowed_method=False)

  def test_annotated_method_is_allowed(self):
    self.assertIsNotNone(
        CallsMethodBeforeSuperConstructorModule(allowed_method=True))

  @parameterized.parameters("trainable_variables", "variables")
  def test_requests_variables_before_they_exist(self, property_name):
    class MyModule(base.Module):
      pass

    mod = MyModule()
    err = "MyModule.* does not currently contain any {}".format(property_name)
    with self.assertRaisesRegex(ValueError, err):
      getattr(mod, property_name)

  @parameterized.parameters("trainable_variables", "variables")
  def test_allow_empty_variables_instance(self, property_name):
    mod = base.Module()
    mod = base.allow_empty_variables(mod)
    self.assertEmpty(getattr(mod, property_name))

  @parameterized.parameters("trainable_variables", "variables")
  def test_allow_empty_variables_class(self, property_name):
    mod = NeverCreatesVariables()
    self.assertEmpty(getattr(mod, property_name))


class NoopModule(base.Module):

  def __init__(self, a=None):
    super().__init__()
    self.a = a


class RaisesOnEquality:

  equality_checked = False

  def __repr__(self):
    return "hostile"

  def __eq__(self, other):
    self.equality_checked = True
    raise ValueError("== not supported")

  def __ne__(self, other):
    self.equality_checked = True
    raise ValueError("!= not supported")


@base.allow_empty_variables
class NeverCreatesVariables(base.Module):
  pass


class ModuleWithFunctionAnnotatedCall(base.Module):

  @tf.function(autograph=False)
  def forward(self):
    return get_name_scope()

  @tf.function(autograph=True)
  def forward_ag(self):
    return get_name_scope()


class CtorNoNameScope(base.Module):

  @base.no_name_scope
  def __init__(self):
    super().__init__()
    self.ctor_name_scope = get_name_scope()
    self.w = tf.Variable(1., name="w")


class CtorNoNameScopeNoSuper(base.Module):

  @base.no_name_scope
  def __init__(self):
    pass


class PropertyModule(base.Module):

  def __init__(self):
    super().__init__()
    self._setter_scope_name = None

  @property
  def some_property(self):
    getter_scope_name = get_name_scope()
    return getter_scope_name, self._setter_scope_name

  @some_property.setter
  def some_property(self, my_property):
    self._setter_scope_name = get_name_scope()

  @property
  @base.no_name_scope
  def no_name_scope_property(self):
    getter_scope_name = get_name_scope()
    return getter_scope_name, self._setter_scope_name

  @no_name_scope_property.setter
  @base.no_name_scope
  def no_name_scope_property(self, my_property):
    self._setter_scope_name = get_name_scope()


class DoesNotCallSuperConstructorModule(base.Module):

  def __init__(self):
    # NOTE: Intentionally does not call super constructor.
    pass


class CallsMethodBeforeSuperConstructorModule(base.Module):

  def __init__(self, allowed_method):
    if allowed_method:
      self.no_name_scope()
    else:
      self.with_name_scope()
    super().__init__()

  @base.no_name_scope
  def no_name_scope(self):
    pass

  def with_name_scope(self):
    pass


class CustomMetaclass(type):

  TAG = "__custom_metaclass__"

  def __new__(cls, name, bases, clsdict):
    new_type = super(CustomMetaclass, cls).__new__(cls, name, bases, clsdict)
    setattr(new_type, CustomMetaclass.TAG, True)
    return new_type


class CombiningMetaclass(base.ModuleMetaclass, CustomMetaclass):

  TAG = "__combining_metaclass__"

  def __new__(cls, name, bases, clsdict):
    new_type = super(CombiningMetaclass, cls).__new__(cls, name, bases, clsdict)  # pylint: disable=too-many-function-args
    setattr(new_type, CombiningMetaclass.TAG, True)
    return new_type


class ModuleWithCustomMetaclass(base.Module, metaclass=CombiningMetaclass):

  def __init__(self):
    super(ModuleWithCustomMetaclass, self).__init__()
    self.init_name_scope = get_name_scope()


class CustomMetaclassTest(tf.test.TestCase):

  def testSupportsCustomMetaclass(self):
    m = ModuleWithCustomMetaclass()
    self.assertEqual(m.init_name_scope, "module_with_custom_metaclass/")
    self.assertTrue(getattr(ModuleWithCustomMetaclass, CombiningMetaclass.TAG))
    self.assertTrue(getattr(ModuleWithCustomMetaclass, CustomMetaclass.TAG))


class TakesSubmodules(base.Module):

  def __init__(self, submodules, name=None):
    super().__init__(name=name)


class WildcardInit(base.Module):

  def __init__(self, a, b, *args, **kwargs):
    super().__init__()
    del args, kwargs


if __name__ == "__main__":
  tf.test.main()
