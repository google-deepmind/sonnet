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
    parent_scope = super(TpuReplicator, self).scope()
    with parent_scope, tf.variable_creator_scope(replica_local_creator_tpu):
      yield
