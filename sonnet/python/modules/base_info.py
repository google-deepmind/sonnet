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
"""Sonnet module information, stored in the graph collections."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
# Dependency imports
import six
from sonnet.protos import module_pb2
from sonnet.python.modules import base_errors
import tensorflow as tf
from tensorflow.python.framework import ops

logging = tf.logging


SONNET_COLLECTION_NAME = "sonnet"


ModuleInfo = collections.namedtuple(
    "ModuleInfo",
    ("module_name", "scope_name", "class_name", "connected_subgraphs"))


ConnectedSubGraph = collections.namedtuple(
    "ConnectedSubGraph", ("module", "name_scope", "inputs", "outputs"))


_SPARSE_TENSOR_NAME = "SparseTensor"
_SPARSE_TENSOR_FIELD = ("indices", "values", "dense_shape")


class _UnserializableObject(object):
  """Placeholder for object which cannot be serialized."""


# Placeholder for tensor which cannot be found.
_MissingTensor = collections.namedtuple("_MissingTensor", ("name",))


def _is_namedtuple(obj):
  """Returns `True` if `obj` is a `collections.namedtuple`."""
  return isinstance(obj, tuple) and hasattr(obj, "_fields")


def _is_iterable(obj):
  """Returns `True` if the object is a supported iterable."""
  return isinstance(obj, (list, tuple, dict))


def _graph_element_to_path(graph_element):
  """Returns the path of the given graph element.

  Args:
    graph_element: A graph element. Currently only `tf.Tensor` is supported.

  Returns:
    The graph path corresponding to `graph_element` or the empty string if no
      path could be found.
  """
  if isinstance(graph_element, tf.Tensor):
    return graph_element.name
  # Returns an empty string when no name is defined. This will be deserialized
  # as a `_UnSerializableObject`.
  return ""


def _path_to_graph_element(path, graph):
  """Returns the graph element of the given path.

  Args:
    path: The path of the graph element.
    graph: The graph to look into.

  Returns:
    The graph element or an instance of `_MissingTensor`.
  """
  try:
    return graph.get_tensor_by_name(path)
  except KeyError:
    return _MissingTensor(path)


def _to_proto_sparse_tensor(sparse_tensor, nested_proto,
                            process_leafs, already_processed):
  """Serializes a `tf.SparseTensor` into `nested_proto`.

  Args:
    sparse_tensor: An instance of `tf.SparseTensor`.
    nested_proto: A `module_pb2.NestedData` instance to be filled from
      `sparse_tensor`.
    process_leafs: A function to be applied to the leaf valued of the nested
      structure.
    already_processed: Set of already processed objects (used to avoid
      infinite recursion).
  """
  already_processed.add(id(sparse_tensor))
  nested_proto.named_tuple.name = _SPARSE_TENSOR_NAME
  for str_key in _SPARSE_TENSOR_FIELD:
    tensor = getattr(sparse_tensor, str_key)
    nested_proto.named_tuple.map[str_key].value = process_leafs(tensor)


def _from_proto_sparse_tensor(sparse_tensor_proto, process_leafs):
  """Deserializes a `tf.SparseTensor` from `sparse_tensor_proto`.

  Args:
    sparse_tensor_proto: A proto representing a `tf.SparseTensor`.
    process_leafs: A function to be applied to the leaf valued of the nested
      structure.

  Returns:
    An instance of `tf.SparseTensor`.
  """
  if not sparse_tensor_proto.HasField("named_tuple"):
    raise base_errors.ModuleInfoError(
        "Error while deserializing a SparseTensor: expected proto tuple.")
  if sparse_tensor_proto.named_tuple.name != _SPARSE_TENSOR_NAME:
    raise base_errors.ModuleInfoError(
        "Error while deserializing a SparseTensor: The name of the tuple "
        "should have been {} but was {}.".format(
            _SPARSE_TENSOR_NAME, sparse_tensor_proto.named_tuple.name))
  named_tuple_map = sparse_tensor_proto.named_tuple.map
  return tf.SparseTensor(
      indices=process_leafs(named_tuple_map["indices"].value),
      values=process_leafs(named_tuple_map["values"].value),
      dense_shape=process_leafs(named_tuple_map["dense_shape"].value))


# This named tuple contains the necessary information to handle a Python
# object which should be handled in a specific way. The "check" field should
# contain a callable returning `True` if the Python object is indeed special
# and the "to_proto" field should contain a custom serializer.
_SpecialTypeInfo = collections.namedtuple("_SpecialTypeInfo",
                                          ("check", "to_proto", "from_proto"))


_TO_PROTO_SPECIAL_TYPES = collections.OrderedDict()
_TO_PROTO_SPECIAL_TYPES[_SPARSE_TENSOR_NAME] = _SpecialTypeInfo(
    check=lambda obj: isinstance(obj, tf.SparseTensor),
    to_proto=_to_proto_sparse_tensor,
    from_proto=_from_proto_sparse_tensor)


def _nested_to_proto(nested_value, nested_proto, process_leafs,
                     already_processed):
  """Serializes `nested_value` into `nested_proto`.

  Args:
    nested_value: A nested Python value.
    nested_proto: A `module_pb2.NestedData` instance to be filled from the value
      in `nested_value`.
    process_leafs: A function to be applied to the leaf values of the nested
      structure.
    already_processed: Set of already processed objects (used to avoid
      infinite recursion).
  Raises:
    ModuleInfoError: If `nested_proto` is not an instance of
      `module_pb2.NestedData`.
  """
  if not isinstance(nested_proto, module_pb2.NestedData):
    raise base_errors.ModuleInfoError("Expected module_pb2.NestedData.")

  # If this object was already processed, mark as "unserializable"
  # to avoid infinite recursion.
  if id(nested_value) in already_processed:
    nested_proto.value = ""
    return

  # Check special types.
  for type_name, type_info in six.iteritems(_TO_PROTO_SPECIAL_TYPES):
    if type_info.check(nested_value):
      nested_proto.special_type.name = type_name
      type_info.to_proto(
          nested_value, nested_proto.special_type.object,
          process_leafs, already_processed)
      return

  # Check standard types.
  if _is_iterable(nested_value):
    # Mark this container as "already processed" to avoid infinite recursion.
    already_processed.add(id(nested_value))
    if isinstance(nested_value, dict):
      nested_proto.dict.SetInParent()
      for key, child in six.iteritems(nested_value):
        str_key = str(key)
        child_proto = nested_proto.dict.map[str_key]
        _nested_to_proto(child, child_proto, process_leafs, already_processed)
    elif isinstance(nested_value, tuple):
      # NamedTuple?
      if _is_namedtuple(nested_value):
        nested_proto.named_tuple.name = type(nested_value).__name__
        for str_key in nested_value._fields:
          child = getattr(nested_value, str_key)
          child_proto = nested_proto.named_tuple.map[str_key]
          _nested_to_proto(child, child_proto, process_leafs, already_processed)
      else:
        nested_proto.tuple.SetInParent()
        for child in nested_value:
          child_proto = nested_proto.tuple.list.add()
          _nested_to_proto(child, child_proto, process_leafs, already_processed)
    else:
      nested_proto.list.SetInParent()
      for child in nested_value:
        child_proto = nested_proto.list.list.add()
        _nested_to_proto(child, child_proto, process_leafs, already_processed)
  else:
    nested_proto.value = process_leafs(nested_value)


def _module_info_to_proto(module_info, export_scope=None):
  """Serializes `module_into`.

  Args:
    module_info: An instance of `ModuleInfo`.
    export_scope: Optional `string`. Name scope to remove.

  Returns:
    An instance of `module_pb2.SonnetModule`.
  """
  def strip_name_scope(name_scope):
    return ops.strip_name_scope(name_scope, export_scope)
  def process_leafs(value):
    return strip_name_scope(_graph_element_to_path(value))
  module_info_def = module_pb2.SonnetModule(
      module_name=module_info.module_name,
      scope_name=strip_name_scope(module_info.scope_name),
      class_name=module_info.class_name)
  for connected_subgraph in module_info.connected_subgraphs:
    connected_subgraph_info_def = module_info_def.connected_subgraphs.add()
    connected_subgraph_info_def.name_scope = strip_name_scope(
        connected_subgraph.name_scope)
    _nested_to_proto(
        connected_subgraph.inputs,
        connected_subgraph_info_def.inputs,
        process_leafs, set())
    _nested_to_proto(
        connected_subgraph.outputs,
        connected_subgraph_info_def.outputs,
        process_leafs, set())
  return module_info_def


def _nested_from_proto(nested_proto, process_leafs):
  """Deserializes `nested_proto`.

  Args:
    nested_proto: An instance of `module_pb2.NestedData`.
    process_leafs: A function to be applied to the leaf values of the nested
      structure.

  Returns:
    An instance of `string`, `tuple`, `dict` or `namedtuple`.

  Raises:
    base_errors.ModuleInfoError: If the probobuf is of the wrong type or
      if some of its fields are missing.
  """
  if not isinstance(nested_proto, module_pb2.NestedData):
    raise base_errors.ModuleInfoError("Expected module_pb2.NestedData.")

  if nested_proto.HasField("value"):
    value = nested_proto.value
    if not value:
      value = _UnserializableObject()
    else:
      value = process_leafs(value)
    return value
  elif nested_proto.HasField("list"):
    return [_nested_from_proto(child, process_leafs)
            for child in nested_proto.list.list]
  elif nested_proto.HasField("tuple"):
    return tuple(_nested_from_proto(child, process_leafs)
                 for child in nested_proto.tuple.list)
  elif nested_proto.HasField("dict"):
    return {name: _nested_from_proto(child, process_leafs)
            for name, child in six.iteritems(nested_proto.dict.map)}
  elif nested_proto.HasField("named_tuple"):
    tmp_dict = {name: _nested_from_proto(child, process_leafs)
                for name, child in six.iteritems(nested_proto.named_tuple.map)}
    # Note that this needs to be a named tuple to work with existing usage.
    NamedTuple = collections.namedtuple(  # pylint: disable=invalid-name
        nested_proto.named_tuple.name, tmp_dict.keys())
    return NamedTuple(**tmp_dict)
  elif nested_proto.HasField("special_type"):
    if nested_proto.special_type.name not in _TO_PROTO_SPECIAL_TYPES:
      return _UnserializableObject()
    type_info = _TO_PROTO_SPECIAL_TYPES[nested_proto.special_type.name]
    return type_info.from_proto(nested_proto.special_type.object, process_leafs)
  else:
    raise base_errors.ModuleInfoError(
        "Cannot deserialize a `ModuleInfo` protobuf with no fields.")


def _module_info_from_proto(module_info_def, import_scope=None):
  """Deserializes `module_info_def` proto.

  Args:
    module_info_def: An instance of `module_pb2.SonnetModule`.
    import_scope: Optional `string`. Name scope to use.

  Returns:
    An instance of `ModuleInfo`.

  Raises:
    base_errors.ModuleInfoError: If the probobuf is of the wrong type or
      if some of its fields are missing.
  """
  graph = tf.get_default_graph()
  def prepend_name_scope(name_scope):
    return ops.prepend_name_scope(name_scope, import_scope)
  def process_leafs(name):
    return _path_to_graph_element(prepend_name_scope(name), graph)
  connected_subgraphs = []
  module_info = ModuleInfo(
      module_name=module_info_def.module_name,
      scope_name=prepend_name_scope(module_info_def.scope_name),
      class_name=module_info_def.class_name,
      connected_subgraphs=connected_subgraphs)
  for connected_subgraph_def in module_info_def.connected_subgraphs:
    connected_subgraph = ConnectedSubGraph(
        module=module_info,
        name_scope=prepend_name_scope(connected_subgraph_def.name_scope),
        inputs=_nested_from_proto(
            connected_subgraph_def.inputs, process_leafs),
        outputs=_nested_from_proto(
            connected_subgraph_def.outputs, process_leafs))
    connected_subgraphs.append(connected_subgraph)
  return module_info


def _module_info_from_proto_safe(module_info_def, import_scope=None):
  """Deserializes the `module_info_def` proto without raising exceptions.

  Args:
    module_info_def: An instance of `module_pb2.SonnetModule`.
    import_scope: Optional `string`. Name scope to use.

  Returns:
    An instance of `ModuleInfo`.
  """
  try:
    return _module_info_from_proto(module_info_def, import_scope)
  except Exception as e:  # pylint: disable=broad-except
    logging.warning(
        "Error encountered when deserializing sonnet ModuleInfo:\n%s", str(e))
    return None


# `to_proto` is already wrapped into a try...except externally but
# `from_proto` isn't. In order to minimize disruption, catch all the exceptions
# happening during `from_proto` and just log them.
ops.register_proto_function(SONNET_COLLECTION_NAME,
                            module_pb2.SonnetModule,
                            to_proto=_module_info_to_proto,
                            from_proto=_module_info_from_proto_safe)
