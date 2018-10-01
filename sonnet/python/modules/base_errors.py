# Copyright 2017 The Sonnet Authors. All Rights Reserved.
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
"""Sonnet exception classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Error(Exception):
  """Base class for all errors from snt.

  This is thrown to indicate a Neural Network specific problem, e.g. wrong
  module arity, module is not connected to the graph when it should be,
  tried to wire together incompatible modules, etc.
  """


class NotConnectedError(Error):
  """Error raised when operating on a module that has not yet been connected.

  Some module properties / methods are valid to access before the module has
  been connected into the graph, but some are not. This Error is raised when
  the user attempts to do anything not valid before connection.
  """


class ParentNotBuiltError(Error):
  """Error raised when the parent of a module has not been built yet.

  For example, when making a transpose of modules that inherit from
  `module.Transposable`, the parent has to be connected to the graph before the
  child transpose to ensure that shape inference has already occurred.
  """


class IncompatibleShapeError(Error):
  """Error raised when the shape of the input at build time is incompatible."""


class UnderspecifiedError(Error):
  """Error raised when too little information is available.

  This does not typically mean the user is trying to do something that doesn't
  work (in which case `IncompatibleShapeError` should be used), just that
  some more information needs to be provided in order to build the Graph.
  """


class NotSupportedError(Error):
  """Error raised when something that cannot be supported is requested.

  For example a Dilated Convolution module cannot be transposed.
  """


class NotInitializedError(Error):
  """Error raised when connecting an uninitialized Sonnet module.

  Before they can be connected, all Sonnet modules must call
  `AbstractModule.__init__` (e.g. via a `super` call).
  """


class DifferentGraphError(Error):
  """Error raised when trying to connect a Sonnet module to multiple Graphs."""


class ModuleInfoError(Error):
  """Error raised when Sonnet `ModuleInfo` cannot be serialized."""
