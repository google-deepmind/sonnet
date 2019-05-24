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
from __future__ import print_function

import contextlib
import tensorflow as tf


def _replica_local_creator(getter, **kwargs):  # pylint: disable=missing-docstring
  if kwargs["synchronization"] == tf.VariableSynchronization.AUTO:
    kwargs["synchronization"] = tf.VariableSynchronization.ON_READ
    if kwargs["aggregation"] == tf.VariableAggregation.NONE:
      kwargs["aggregation"] = tf.VariableAggregation.ONLY_FIRST_REPLICA
  v = getter(**kwargs)
  # TODO(petebu): Revisit this once ReplicaLocalVariable allows trainable=True.
  if kwargs["trainable"] is None and not v.trainable:
    for c in v._values:  # pylint: disable=protected-access
      c._trainable = True  # pylint: disable=protected-access
  return v


class Replicator(tf.distribute.MirroredStrategy):
  """Replicator Distribution Strategy.

  A distribution strategy which creates ReplicaLocal variables by default. These
  are not automatically aggregated in a replica context. Instead, you must
  manually aggregate them e.g. using `tf.distribute.ReplicaContext.all_reduce`.
  In a cross-replica context, reading reads from the first replica only and
  assignment broadcasts to all replicas.
  """

  @contextlib.contextmanager
  def scope(self):
    parent_scope = super(Replicator, self).scope()
    with parent_scope, tf.variable_creator_scope(_replica_local_creator):
      yield
