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

from typing import Callable, TypeVar

from absl import logging
import contextlib
from sonnet.src import initializers
import tensorflow as tf

T = TypeVar("T")


def replica_local_creator(next_creator, **kwargs) -> tf.Variable:
  """Variable creator that by default creates replica local variables."""
  if kwargs["synchronization"] == tf.VariableSynchronization.AUTO:
    kwargs["synchronization"] = tf.VariableSynchronization.ON_READ
    if kwargs["aggregation"] == tf.VariableAggregation.NONE:
      kwargs["aggregation"] = tf.VariableAggregation.ONLY_FIRST_REPLICA
    if kwargs["trainable"] is None:
      kwargs["trainable"] = True
  return next_creator(**kwargs)


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

      >>> per_replica_y = replicator.run(forward)
  """

  @contextlib.contextmanager
  def scope(self):
    with contextlib.ExitStack() as stack:
      stack.enter_context(super().scope())
      stack.enter_context(tf.variable_creator_scope(replica_local_creator))
      yield

# TODO(tomhennigan) Remove this once TF 2.3 is released.
try:
  TPUStrategy = tf.distribute.TPUStrategy
except AttributeError:
  TPUStrategy = tf.distribute.experimental.TPUStrategy


class TpuReplicator(TPUStrategy):
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
      ...   return replicator.run(forward)
      >>> per_replica_y = all_forward()
  """

  @contextlib.contextmanager
  def scope(self):
    with contextlib.ExitStack() as stack:
      stack.enter_context(super().scope())
      stack.enter_context(tf.variable_creator_scope(replica_local_creator))
      yield


def create_variables_eagerly(f: Callable[..., T]) -> Callable[..., T]:
  """Wraps a function and attempts to create variables using eager mode.

  Example usage:

  >>> model = snt.Sequential([snt.Linear(1) for _ in range(100)])

  >>> @tf.function
  ... @snt.distribute.create_variables_eagerly
  ... def f(x):
  ...   return model(x)

  >>> _ = f(tf.ones([1, 1]))

  On a CPU only machine ``f`` will run ~4x faster (700ms vs. 3s), the benefits
  are more pronounced in a distributed setup since eager variable creation can
  skip a number of checks that are required in graph mode (e.g. checking whether
  the variable has already been created) which end up ping-ponging RPCs.

  Args:
    f: Any function.

  Returns:
    A function running `f` in a context where variables are created eagerly.
  """
  def wrapper(*args, **kwargs):
    with contextlib.ExitStack() as stack:
      # The two hacks below enable a large speedup when initializing large
      # models on TPU pods.
      # TODO(b/141243467) Remove these workarounds.
      stack.enter_context(_eager_initial_values())
      stack.enter_context(tf.variable_creator_scope(_eager_variable_creator))
      return f(*args, **kwargs)
  return wrapper


def _eager_variable_creator(getter, initial_value, **kwargs):
  """Attempts to force variable creation to be eager."""
  eager_initial_value = None

  if isinstance(initial_value, tf.Tensor):
    eager_initial_value = tf.get_static_value(initial_value)

  if eager_initial_value is not None:
    # If we have an eager initial value we can create variables in eager mode.
    with tf.init_scope():
      return getter(initial_value=eager_initial_value, **kwargs)

  else:
    # Fall back to creating in whatever context we're in with user input.
    return getter(initial_value=initial_value, **kwargs)


@contextlib.contextmanager
def _eager_initial_values():
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
