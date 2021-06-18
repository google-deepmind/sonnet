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
"""Read modules."""

import tensorflow as tf


def read(memory,
         weights,
         squash_op=tf.nn.tanh,
         squash_before_access=True,
         squash_after_access=False):
  """Read from the NTM memory.

  Args:
    memory: 3D Tensor [batch_size, memory_size, word_size].
    weights: 3D Tensor [batch_size, num_reads, memory_size].
    squash_op: op to perform squashing of memory or read word.
    squash_before_access: squash memory before read, default True.
    squash_after_access: squash read word, default False.

  Returns:
    3D Tensor [batch_size, num_reads, word_size].
  """
  with tf.name_scope("read_memory"):
    if squash_before_access:
      squash_op(weights)
    read_word = tf.matmul(weights, memory)
    if squash_after_access:
      read_word = squash_op(read_word)
    return read_word
