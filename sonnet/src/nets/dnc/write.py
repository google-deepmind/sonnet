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
"""Write modules."""

import tensorflow as tf


def additive_write(memory, address, values):
  """Additively writes values to memory at given address.

  M_t = M_{t-1} + w_t a_t^T.

  Args:
    memory: 3D Tensor [batch_size, memory_size, word_size].
    address: 3D Tensor [batch_size, num_writes, memory_size].
    values: 3D Tensor [batch_size, num_writes, word_size].

  Returns:
    3D Tensor [batch_size, num_reads, word_size].
  """
  with tf.name_scope('write_memory'):
    add_matrix = tf.matmul(address, values, adjoint_a=True)
    return memory + add_matrix


def erase(memory, address, reset_weights):
  """Erases rows over addressing distribution by given reset word weights.

  M_t(i) = M_{t-1}(i) * (1 - w_t(i) * e_t)

  The erase is defined as a component-wise OR over reset strengths between
  write heads, followed by a componentwise multiplication. The reset weights
  contains values in [0, 1] where 1 indicates a complete reset. The reset
  weights are granular over a word, allowing for part of the word to be erased.

  Args:
    memory: 3D Tensor [batch_size, memory_size, word_size].
    address: 3D Tensor [batch_size, num_writes, memory_size].
    reset_weights: 3D Tensor [batch_size, num_writes, word_size].

  Returns:
    erased memory: 3D Tensor [batch_size, num_reads, word_size].
  """

  with tf.name_scope('erase_memory'):
    address = tf.expand_dims(address, 3)
    reset_weights = tf.expand_dims(reset_weights, 2)
    weighted_resets = address * reset_weights
    reset_gate = tf.reduce_prod(1 - weighted_resets, [1])
    return memory * reset_gate


def erase_rows(memory, address, reset_row_weights):
  """Erases rows over addressing distribution by given reset weight.

  The reset row weight here is uniform over the values in a word.

  Args:
    memory: 3D Tensor [batch_size, memory_size, word_size].
    address: 3D Tensor [batch_size, num_writes, memory_size].
    reset_row_weights: 2D Tensor [batch_size, num_writes].

  Returns:
    3d Tensor of memory [batch_size, memory_size, word_size].
  """

  with tf.name_scope('erase_rows'):
    # Expands reset_row_weights for broadcasted cmul with address.
    reset_row_weights = tf.expand_dims(reset_row_weights, -1)
    weighted_resets = tf.multiply(address, reset_row_weights)
    reset_gate = tf.reduce_prod(1 - weighted_resets, axis=[1])
    # Expands reset_gate for broadcasted cmul with memory.
    reset_gate = tf.expand_dims(reset_gate, -1)
    return tf.multiply(memory, reset_gate)


def erase_and_write(memory, address, reset_weights, values):
  """Module to erase and write in the NTM memory.

  Implementation is based on equations (3) and (4) from 'Neural Turing Machines'
  (https://arxiv.org/pdf/1410.5401.pdf) by gravesa@, gregwayne@ and danihelka@:

  Erase operation:
    M_t'(i) = M_{t-1}(i) * (1 - w_t(i) * e_t)

  Add operation:
    M_t(i) = M_t'(i) + w_t(i) * a_t

  where e are the reset_weights, w the write weights and a the values.

  Args:
    memory: 3D Tensor [batch_size, memory_size, word_size].
    address: 3D Tensor [batch_size, num_writes, memory_size].
    reset_weights: 3D Tensor [batch_size, num_writes, word_size].
    values: 3D Tensor [batch_size, num_writes, word_size].

  Returns:
    3D Tensor [batch_size, num_reads, word_size].
  """
  memory = erase(memory, address, reset_weights)
  memory = additive_write(memory, address, values)
  return memory
