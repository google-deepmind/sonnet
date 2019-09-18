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

from absl import logging
import contextlib
from sonnet.src import initializers
import tensorflow as tf


def make_replica_local_creator(tpu_strategy):
  """Makes a variable creator which creates replica local variables."""

  def _replica_local_creator(next_creator, **kwargs) -> tf.Variable:
    """Variable creator that by default creates replica local variables."""
    if kwargs["synchronization"] == tf.VariableSynchronization.AUTO:
      if not tpu_strategy:
        # TODO(b/138116230) remove this when sync on read variables work on TPU.
        kwargs["synchronization"] = tf.VariableSynchronization.ON_READ
      if kwargs["aggregation"] == tf.VariableAggregation.NONE:
        kwargs["aggregation"] = tf.VariableAggregation.ONLY_FIRST_REPLICA
      if kwargs["trainable"] is None:
        kwargs["trainable"] = True
    return next_creator(**kwargs)
  return _replica_local_creator


def _is_eager_tensor(t: tf.Tensor):
  try:
    t.op  # pylint: disable=pointless-statement
    return False
  except:  # pylint: disable=bare-except
    return True


def create_variables_eagerly(getter, initial_value, **kwargs):
  """Attempts to force variable creation to be eager."""
  eager_initial_value = None

  if isinstance(initial_value, tf.Tensor):
    if _is_eager_tensor(initial_value):
      eager_initial_value = initial_value
    else:
      # Try to compute the static value (e.g. if the user used `tf.ones`).
      eager_initial_value = tf.get_static_value(initial_value)

  if eager_initial_value is not None:
    # If we have an eager initial value we can create variables in eager mode.
    with tf.init_scope():
      return getter(initial_value=eager_initial_value, **kwargs)

  else:
    # Fall back to creating in whatever context we're in with user input.
    return getter(initial_value=initial_value, **kwargs)


@contextlib.contextmanager
def eager_initial_values():
  """Attempts to force all initializers to create eager tensors."""
  all_initializers = {cls: cls.__call__
                      for cls in initializers.Initializer.__subclasses__()}

  def patched_call(self, shape, dtype):
    """Monkey-patched verison of `Initializer.__call__`."""
    cls = type(self)
    orig_call = all_initializers[cls]
    try:
      with tf.init_scope():
        return orig_call(self, shape, dtype)
    except:  # pylint: disable=bare-except
      if not tf.executing_eagerly():
        logging.exception(
            "Failed to create initial value eagerly for %s shape=%s dtype=%s",
            type(self).__name__, shape, dtype)
      return orig_call(self, shape, dtype)

  try:
    for cls in all_initializers:
      cls.__call__ = patched_call
    yield

  finally:
    # Restore
    for cls, orig_call in all_initializers.items():
      cls.__call__ = orig_call

replica_local_creator = make_replica_local_creator(tpu_strategy=False)
replica_local_creator_tpu = make_replica_local_creator(tpu_strategy=True)


class Replicator(tf.distribute.MirroredStrategy):
  r"""Replicates input, parameters and compute over multiple accelerators.

  ``Replicator`` is a TensorFlow "Distribution Strategy" implementing the
  programming model described in the TF-Replicator paper
  :cite:`buchlovsky2019tf` and TensorFlow RFC
  :cite:`buchlovsky2019distribution`. ``Replicator`` enables data-parallel
  training across multiple accelerators on a single machine, it supports
  eager execution and :tf:`function`.

  To get started create a ``Replicator`` instance:

      >>> replicator = snt.distribute.Replicator()

  Replicator provides a scope inside which any new :tf:`Variable`\ s will be
  replicated across all local devices:

      >>> with replicator.scope():
      ...    mod = snt.Linear(32)

  Additionally replicator provides utility functions to apply a module in
  parallel on multiple devices. First we need to define some computation that
  runs on each GPU. The "replica context" object provides us a way to
  communicate between replicas (e.g. to perform an ``all_reduce``):

      >>> def forward():
      ...   # Compute a random output on each GPU.
      ...   x = tf.random.normal([8, 28 * 28])
      ...   y = mod(x)
      ...   # Synchronize the value of `y` between all GPUs.
      ...   ctx = tf.distribute.get_replica_context()
      ...   y = ctx.all_reduce("mean", y)
      ...   return y

  Finally we use the run API to apply ``forward`` in parallel on all accelerator
  devices:

      >>> per_replica_y = replicator.experimental_run_v2(forward)
  """

  @contextlib.contextmanager
  def scope(self):
    parent_scope = super(Replicator, self).scope()
    with parent_scope, tf.variable_creator_scope(replica_local_creator):
      yield


class TpuReplicator(tf.distribute.experimental.TPUStrategy):
  r"""Replicates input, parameters and compute over multiple TPUs.

  ``TpuReplicator`` is a TensorFlow "Distribution Strategy" implementing the
  programming model described in the TF-Replicator paper
  :cite:`buchlovsky2019tf` and TensorFlow RFC
  :cite:`buchlovsky2019distribution`. ``TpuReplicator`` enables data-parallel
  training across multiple TPUs on one or more machines, it supports
  :tf:`function`.

  To get started create a ``TpuReplicator`` instance:

      >>> replicator = snt.distribute.TpuReplicator()

  This provides a scope inside which any new :tf:`Variable`\ s will be
  replicated across all TPU cores:

      >>> with replicator.scope():
      ...    mod = snt.Linear(32)

  Additionally replicator provides utility functions to apply a module in
  parallel on multiple devices. First we need to define some computation that
  runs on each TPU. The "replica context" object provides us a way to
  communicate between replicas:

      >>> def forward():
      ...   # Compute a random output on each GPU.
      ...   x = tf.random.normal([8, 28 * 28])
      ...   y = mod(x)
      ...   # Synchronize the value of `y` between all GPUs.
      ...   ctx = tf.distribute.get_replica_context()
      ...   y = ctx.all_reduce("mean", y)
      ...   return y

  Finally we use the run API to apply ``forward`` in parallel on all TPU
  devices. This must be run as part of a :tf:`function` since ``TpuReplicator``
  uses XLA to compile and replicate our function to run in parallel over all
  TPU cores:

      >>> @tf.function(autograph=False)
      ... def all_forward():
      ...   return replicator.experimental_run_v2(forward)
      >>> per_replica_y = all_forward()
  """

  @contextlib.contextmanager
  def scope(self):
    with contextlib.ExitStack() as stack:
      stack.enter_context(super(TpuReplicator, self).scope())
      stack.enter_context(tf.variable_creator_scope(replica_local_creator_tpu))

      # The two hacks below enable a large speedup when initializing TPUs (on
      # a 4x4 slice startup for ResNet50 goes from 42m -> 2m).
      # TODO(tomhennigan) Remove these workarounds.
      stack.enter_context(tf.variable_creator_scope(create_variables_eagerly))
      stack.enter_context(eager_initial_values())

      yield
