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

"""Replicator Distribution Strategy."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import contextlib
import functools

import tensorflow as tf


@contextlib.contextmanager
def maybe_enter_scope(strategy):
  """Enter the strategy scope if it is not already active."""
  if strategy is not tf.distribute.get_strategy():
    with strategy.scope():
      yield
  else:
    yield


def replica_local_assign(v, assign_fn):
  """Replaces `assign_fn` on `v` so that it works in cross-replica context."""
  @functools.wraps(v.assign)
  def wrapper(value):
    with maybe_enter_scope(v.distribute_strategy):
      ctx = tf.distribute.get_replica_context()
      if ctx is None:
        for component in v._values:  # pylint: disable=protected-access
          getattr(component, assign_fn)(value)
        return None
      else:
        return getattr(v.get(), assign_fn)(value)
  return wrapper


def replica_local_read_value(v):
  """Replaces `read_value` on `v` so that it works in cross-replica context."""
  @functools.wraps(v.read_value)
  def wrapper():
    with maybe_enter_scope(v.distribute_strategy):
      ctx = tf.distribute.get_replica_context()
      if ctx is None:
        return v._values[0].read_value()  # pylint: disable=protected-access
      else:
        return v.get().read_value()
  return wrapper


def replica_local_creator(getter, **kwargs) -> tf.Variable:
  """Variable creator that by default creates replica local variables."""
  if kwargs["synchronization"] == tf.VariableSynchronization.AUTO:
    kwargs["synchronization"] = tf.VariableSynchronization.ON_READ
    if kwargs["aggregation"] == tf.VariableAggregation.NONE:
      kwargs["aggregation"] = tf.VariableAggregation.ONLY_FIRST_REPLICA
    if kwargs["trainable"] is None:
      kwargs["trainable"] = True
    v = getter(**kwargs)

    # TODO(petebu): Remove when local variables support cross-replica assign.
    v.assign = replica_local_assign(v, "assign")
    v.assign_add = replica_local_assign(v, "assign_add")
    v.assign_sub = replica_local_assign(v, "assign_sub")
    v.read_value = replica_local_read_value(v)
  else:
    v = getter(**kwargs)
  return v


class Replicator(tf.distribute.MirroredStrategy):
  """Replicates input, parameters and compute over multiple accelerators.

  `Replicator` is a TensorFlow "Distribution Strategy" implementing the
  programming model described in the TF-Replicator paper
  :cite:`buchlovsky2019tf` and TensorFlow RFC
  :cite:`buchlovsky2019distribution`. `Replicator` enables data-parallel
  training across multiple accelerators on a single machine, it supports
  eager execution and `@tf.function`.

  To get started create a `Replicator` instance:

      >>> replicator = snt.distribute.Replicator()

  Replicator provides a scope inside which any new `tf.Variable`s will be
  replicated across all local devices:

      >>> with replicator.scope():
      ...    mod = snt.Linear(32)

  Additionally replicator provides utility functions to apply a module in
  parallel on multiple devices. First we need to define some computation that
  runs on each GPU. The "replica context" object provides us a way to
  communicate between replicas:

      >>> def forward():
      ...   # Compute a random output on each GPU.
      ...   x = tf.random.normal([8, 28 * 28])
      ...   y = mod(x)
      ...   # Synchronize the value of `y` between all GPUs.
      ...   ctx = tf.distribute.get_replica_context()
      ...   y = ctx.all_reduce("mean", y)
      ...   return y

  Finally we use the run API to apply `forward` in parallel on all accelerator
  devices:

      >>> per_replica_y = replicator.experimental_run_v2(forward)
  """

  @contextlib.contextmanager
  def scope(self):
    parent_scope = super(Replicator, self).scope()
    with parent_scope, tf.variable_creator_scope(replica_local_creator):
      yield
