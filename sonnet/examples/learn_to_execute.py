# Copyright 2018 The Sonnet Authors. All Rights Reserved.
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

"""Learning To Execute Dataset.

Generated sequences of constant time mini-programs.

Modes:

* `TRAIN_COMBINE`: Uses Combined Curriulum for training.
* `TRAIN_MIX`: Uses Mix Curriulum for training.
* `TRAIN_NAIVE`: Uses Naive Curriulum for training.
* `TEST`: Uses Baseline Currculum for testing.

This module defines:

  1. Set of curriculum classes.
  2. Set of execution operation classes (this may be extended).
  3. Methods for defining a vocabulary and code samples.
  4. Class responsible for handling the tokenization of samples.
  5. Dataset class to generate train/test batches.

This dataset is generative and does not rely on any statically stored data.
Therefore there is no limit to the samples generated. A generated batch will
be of dimensionality [sequence, length, one_hot_encoding_size].  Finally, the
dataset requires a maximum literal length and nesting level, for example:

  (25 if 10 < 2 else (333 - (22 + 4)))

This has a maximum literal length of 3 (`333`) and nesting level of 3
(`(22 + 4)`).

Finally, it should be mentioned that the dataset can operate also in two
tokenization modes: tokenied and detokenized.  In a detokenized mode the
sequence is tokenized by character while in the tokenized mode the sample
is tokenized by keywords, numbers and literals (illustrated by the spaces in
the above example).  This can set with the `token_by_char` arg where
detokenized corresponds to `True`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import random
from enum import Enum
import numpy as np
import six
import tensorflow as tf

_SOS = "~"  # Start of sequence symbol
_EOS = "!"  # End of sequence symbol
_PAD = "."  # Padding symbol

DEFAULT_MIN_CURRICULUM_EVAL_TRIES = 10  # Minimum # of times to attempt update.


@six.add_metaclass(abc.ABCMeta)
class LTECurriculum(object):
  """Base class for code operations."""

  MIN_LOSS_WINDOW_SIZE = 5
  MAX_LOSS_WINDOW_SIZE = 10

  def __init__(self, max_length, max_nesting, loss_threshold=0.05,
               min_tries=DEFAULT_MIN_CURRICULUM_EVAL_TRIES):
    """Initializes the curriculum.

    Args:
      max_length: The maximum literal length.
      max_nesting: The maximum nesting.
      loss_threshold: Fractional value under which the average difference in
        validation loss will trigger additional difficulty.
      min_tries: the minimum number of times required on a difficulty level.
    """
    self._curr_length = 1
    self._max_length = max_length
    self._curr_nesting = 1
    self._max_nesting = max_nesting
    self._loss_threshold = loss_threshold
    self._min_tries = min_tries
    self._curr_tries = 0
    self._set_loss_window()

  def _set_loss_window(self):
    """Initializes the queue that stores the losses."""
    avg_window_size = max(min(self._min_tries, self.MAX_LOSS_WINDOW_SIZE),
                          self.MIN_LOSS_WINDOW_SIZE)
    self._losses = collections.deque(
        [], maxlen=avg_window_size)  # pytype: disable=wrong-arg-count

  @property
  def friendly_name(self):
    return "Root(" + str(self._loss_threshold) + ")"

  def update(self, loss, force=False):
    """Determines whether task level difficulty is to be increased.

    Collects loss values and difference since the last update.  This is used
    to compute the fractional difference across the loss window.

    Args:
      loss: float indicating an loss value to determine whether to update
        the curriculum state.
      force: boolean that allows us to force a curriculum update.

    Returns:
      True if there was an update.
    """
    if force:
      self._curr_tries = 0
      self._set_loss_window()
      self._set_new_task = True
      return True
    self._losses.append(loss)
    if self._curr_tries < self._min_tries - 1:
      self._curr_tries += 1
      return False

    # Average change in loss normalized by average loss.
    loss_diffs = [pair[0] - pair[1]
                  for pair in zip(list(self._losses)[1:],
                                  list(self._losses)[:-1])]
    avg_loss_norm = np.mean(loss_diffs) / np.mean(self._losses)
    if avg_loss_norm < self._loss_threshold:
      self._set_new_task = True
      self._curr_tries = 0
      self._set_loss_window()
      return True
    else:
      return False

  def fetch(self):
    """Getter for current curriculum nesting and length.

    Returns:
      Tuple of integer values indicating literal length and nesting depth.
    """
    return self._curr_length, self._curr_nesting

  @property
  def current_level(self):
    """Gets current currciculum level (string)."""
    return str(self._curr_nesting) + "." + str(self._curr_length)

  @property
  def max_length(self):
    """Gets maximum literal depth."""
    return self._max_length

  @property
  def max_nesting(self):
    """Gets maximum nesting depth."""
    return self._max_nesting


class BaselineCurriculum(LTECurriculum):
  """Baseline curriculum sets a fixed nesting and length."""

  def __init__(self, length, nesting, threshold,
               min_tries=DEFAULT_MIN_CURRICULUM_EVAL_TRIES):
    tf.logging.info("Initializing Baseline curriculum. length=%d, nest=%d, "
                    "valid threshold=%f", length, nesting, threshold)
    super(BaselineCurriculum, self).__init__(length, nesting, threshold,
                                             min_tries)
    self._curr_length = length
    self._curr_nesting = nesting

  @property
  def friendly_name(self):
    return "Baseline(" + str(self._loss_threshold) + ")"


class NaiveCurriculum(LTECurriculum):
  """Naive curriculum increments length, nesting complexity by 1 on update."""

  def __init__(self, length, nesting, threshold,
               min_tries=DEFAULT_MIN_CURRICULUM_EVAL_TRIES):
    tf.logging.info("Initializing Naive curriculum."
                    " length=%d, nest=%d, valid threshold=%f", length, nesting,
                    threshold)
    super(NaiveCurriculum, self).__init__(length, nesting, threshold,
                                          min_tries)

  def friendly_name(self):
    return "Naive(" + str(self._loss_threshold) + ")"

  def update(self, loss, force=False):
    """Increments level difficulty (length and nesting) by 1 until maximum."""
    do_update = super(NaiveCurriculum, self).update(loss, force)
    if do_update:
      if self._curr_length < self._max_length:
        self._curr_length += 1
        return True
      elif self._curr_nesting < self._max_nesting:
        self._curr_nesting += 1
      else:
        self._set_new_task = False
      if self._set_new_task:
        tf.logging.info("New level: (length=%d, nesting=%d)",
                        self._curr_length,
                        self._curr_nesting)
      return self._set_new_task
    return False


class MixCurriculum(LTECurriculum):
  """Mixed chooses randomly by batch up to a maximum length/nesting."""

  def __init__(self, length, nesting, threshold,
               min_tries=DEFAULT_MIN_CURRICULUM_EVAL_TRIES):
    tf.logging.info("Initializing Mix curriculum."
                    " length=%d, nest=%d, valid threshold=%f", length, nesting,
                    threshold)
    super(MixCurriculum, self).__init__(length, nesting, threshold, min_tries)

  def friendly_name(self):
    return "Mix(" + str(self._loss_threshold) + ")"

  def fetch(self):
    """Samples up to maximum difficulty."""
    length = np.random.randint(1, self._max_length + 1)
    nesting = np.random.randint(1, self._max_length + 1)
    return length, nesting


class CombineCurriculum(LTECurriculum):
  """Combine uses both Mix and Naive strategy together."""

  def __init__(self, length, nesting, threshold,
               min_tries=DEFAULT_MIN_CURRICULUM_EVAL_TRIES):
    tf.logging.info("Initializing Combine curriculum. length=%d, nest=%d, "
                    "valid threshold=%f", length, nesting, threshold)
    super(CombineCurriculum, self).__init__(length, nesting, threshold,
                                            min_tries)

  @property
  def friendly_name(self):
    return "Combine(" + str(self._loss_threshold) + ")"

  def update(self, loss, force=False):
    """Increments level difficulty (length and nesting) by 1 until maximum."""
    do_update = super(CombineCurriculum, self).update(loss, force)
    if not do_update:
      return False

    if self._curr_length < self._max_length:
      self._curr_length += 1
    elif self._curr_nesting < self._max_nesting:
      self._curr_nesting += 1
    else:
      self._set_new_task = False

    if self._set_new_task:
      tf.logging.info("New level: (length=%d, nesting=%d)",
                      self._curr_length, self._curr_nesting)
    return self._set_new_task

  def fetch(self):
    """Samples up to current difficulty."""
    length = np.random.randint(1, self._curr_length + 1)
    nesting = np.random.randint(1, self._curr_nesting + 1)
    return length, nesting


@six.add_metaclass(abc.ABCMeta)
class CodeOp(object):
  """Base class for code operations."""

  def __init__(self, num_operands):
    """Constructor for base operations class.

    This constructor sets the operands for the operation as well as whether
    the operation is a "memory" operation - that is, one the output is a
    permuation or copy of the input.

    Args:
      num_operands: integer indicating number of operands for this operation.
    """
    self._num_operands = num_operands
    self._is_memory = False

  @property
  def num_operands(self):
    """Property returning integer number of operands for the operation."""
    return self._num_operands

  @property
  def is_memory(self):
    """Property indicating whether this is a memory Operation."""
    return self._is_memory

  KEYWORDS = ["if", "for", "else", "range"]
  LITERALS = ["+", "-", "*", "=", ":", "/", "(", ")", " ", "x", "|", "<", ">",
              "[", "]", _SOS, _EOS, _PAD]

  def check_elems(self, elems, count, elem_type):
    """Ensures element list length and type valid.

    Args:
      elems: list of elements.
      count: how many elements are expected.
      elem_type: type of all elements.

    Raises:
      ValueError: When length and type of elems is not as exepcted.
    """
    if len(elems) != count and all(isinstance(e, elem_type) for e in elems):
      raise ValueError("Not all elements valid: {}".format(elems))

  @abc.abstractmethod
  def eval(self, values):
    """Evaluates the operation based with given values."""
    return

  @abc.abstractmethod
  def get_code(self, codes):
    """Composes the operation code from code components."""
    return


class AddOp(CodeOp):
  """Add operation class."""

  def __init__(self):
    super(AddOp, self).__init__(2)

  def eval(self, values):
    self.check_elems(values, 2, int)
    return values[0] + values[1]

  def get_code(self, codes):
    self.check_elems(codes, 2, str)
    return "+".join(codes[:2])


class SubtractOp(CodeOp):
  """Subtract operation class."""

  def __init__(self):
    super(SubtractOp, self).__init__(2)

  def eval(self, values):
    self.check_elems(values, 2, int)
    return values[0] - values[1]

  def get_code(self, codes):
    self.check_elems(codes, 2, str)
    return "".join([codes[0], "-", codes[1]])


class MultiplyOp(CodeOp):
  """Multiply operation class."""

  def __init__(self):
    super(MultiplyOp, self).__init__(2)

  def eval(self, values):
    self.check_elems(values, 2, int)
    return values[0] * values[1]

  def get_code(self, codes):
    self.check_elems(codes, 2, str)
    return "".join([codes[0], "*", codes[1]])


class DivideOp(CodeOp):
  """Divide operation class."""

  def __init__(self):
    super(DivideOp, self).__init__(2)

  def eval(self, values):
    self.check_elems(values, 2, int)
    return values[0] / values[1]

  def get_code(self, codes):
    self.check_elems(codes, 2, str)
    return "".join([codes[0], "/", codes[1]])


class IfOp(CodeOp):
  """If operation class."""

  def __init__(self):
    super(IfOp, self).__init__(4)
    self._comparators = ["<", ">"]

  def eval(self, values):
    self.check_elems(values, 4, int)
    comparator_idx = random.randint(0, len(self._comparators)-1)
    self._comparator = self._comparators[comparator_idx]
    if self._comparator == ">":
      return values[0] if values[1] > values[2] else values[3]
    elif self._comparator == "<":
      return values[0] if values[1] < values[2] else values[3]
    else:
      ValueError("Invalid comparator.")

  def get_code(self, codes):
    self.check_elems(codes, 4, str)
    if self._comparator == ">":
      return "".join([codes[0], "if", codes[1], ">", codes[2],
                      "else", codes[3]])
    elif self._comparator == "<":
      return "".join([codes[0], "if", codes[1], "<", codes[2],
                      "else", codes[3]])
    else:
      ValueError("Invalid comparator.")


class ForOp(CodeOp):
  """For loop operation class."""

  def __init__(self):
    super(ForOp, self).__init__(2)

  def eval(self, values):
    values = list(values)
    self.check_elems(values, 2, int)
    self._it = random.randint(1, 9)
    for _ in six.moves.range(self._it):
      values[0] += values[1]
    return values[0]

  def get_code(self, codes):
    self.check_elems(codes, 2, str)
    return "".join(["x=", codes[0], "for[" + str(self._it) + "]",
                    "x+=", codes[1]])


class ReverseOp(CodeOp):
  """Outputs a reversal of the input."""

  def __init__(self):
    """Constructor for ReverseOp."""
    super(ReverseOp, self).__init__(1)
    self._is_memory = True

  def eval(self, values):
    """Evaluation method for reverse operation.

    Args:
      values: List of samples to compose operation.

    Returns:
      String representing reversed input.
    """
    return str(values[0])[::-1]

  def get_code(self, codes):
    """Composes a code for double-copy operation.

    Args:
      codes: List of samples to compose operation.

    Returns:
      String for code of reversed input result.
    """
    return "".join(codes)


class CopyOp(CodeOp):
  """Outputs a copy of the input."""

  def __init__(self):
    """Constructor for CopyOp."""
    super(CopyOp, self).__init__(1)
    self._is_memory = True

  def eval(self, values):
    """Evaluation method for copy operation.

    Args:
      values: List of samples to compose operation.

    Returns:
      String representing copied input.
    """
    return values[0]

  def get_code(self, codes):
    """Composes a code for double-copy operation.

    Args:
      codes: List of samples to compose operation code.

    Returns:
      String for code of copied input result.
    """
    return "".join(codes)


class DoubleCopyOp(CodeOp):
  """Outputs two concatenated copies of the input."""

  def __init__(self):
    """Constructor for DoubleCopyOp."""
    super(DoubleCopyOp, self).__init__(1)
    self._is_memory = True

  def eval(self, values):
    """Evaluation method for DoubleCopy operation.

    Args:
      values: List of samples to compose operation code.

    Returns:
      String representing doubled input result.
    """
    return str(values[0]) + str(values[0])

  def get_code(self, codes):
    """Composes a code for double-copy operation.

    Args:
      codes: List of samples to compose operation.

    Returns:
      String for code of double copied input result.
    """
    return "".join(codes)


def generate_code(max_length, max_nest, ops):
  """Generates code samples.

  Args:
    max_length: int.  max literal length.
    max_nest: int. max nesting level.
    ops: CodeOp. set of allowable operations.

  Returns:
    1. (str) output value.
    2. (str) Code operation.
  """
  stack = []
  def fetch_one():
    # Always use an existing nested value for one of the operands.
    if stack:
      return stack.pop()
    else:
      # Produce a numeral of max_length-digits.
      value = random.randint(10 ** (max_length - 1), 10 ** max_length - 1)
      code = str(value)
      return value, code

  def fetch(num_operands):
    values, codes = zip(*[fetch_one() for _ in six.moves.range(num_operands)])
    return values, codes

  for _ in six.moves.range(max_nest):
    op = random.choice(ops)
    values, codes = fetch(op.num_operands)
    new_value = op.eval(values)
    new_code = op.get_code(codes)
    stack.append((new_value, "(" + new_code + ")"))
  final_value, final_code = stack.pop()
  final_code = final_code[1:-1]
  final_code.strip("()")
  if not op.is_memory:
    final_value = int(final_value) % 10 ** (max_length+1)
  return str(final_value), final_code


def get_tokens(max_value):
  """Defines tokens.

  Args:
    max_value: the maximum numeric range for the token.

  Returns:
    list of string tokens in vocabulary.
  """
  vocab = [str(i) for i in range(max_value)]
  vocab = set(vocab)
  vocab.update(CodeOp.LITERALS)
  vocab.update(CodeOp.KEYWORDS)
  vocab |= set("".join(vocab))
  return sorted(vocab)


def get_padding():
  """Returns the padding character."""
  return _PAD


def get_start_token():
  """Returns `start-of-sequence` character."""
  return _SOS


def get_end_token():
  """Returns `end-of-sequence` character."""
  return _EOS


class TokenDataSource(object):
  """Encapsulates loading/tokenization logic for samples from generator."""

  UNK = "_unk_"
  DEFAULT_START_TOKENS = ["_null_", "_eos_", "|"]
  NULL, WORD_EOS, CHAR_EOS = DEFAULT_START_TOKENS

  def __init__(self, curriculum_obj, batch_size, max_len, ops, token_by_char):
    """Creates a TokenDataSource instance.

    Args:
      curriculum_obj: (LTECurriculum) determines sample complexity.
      batch_size: (int) Batch size to generate.
      max_len: (int) This is the maximum size of any given sample sequence.
      ops: (list(CodeOp)). Task operations that inherit from CodeOp().
      token_by_char: (bool) Whether to tokenize by char ("detokenized") or by
          keyword, literals and numbers.
    """
    # Create the token and inverse-token dicts and fix the UNK token.
    self._vocab_dict = collections.defaultdict(lambda: 0)
    self._vocab_dict[self.UNK] = 0
    self._inv_vocab_dict = collections.defaultdict(lambda: self.UNK)

    self.curriculum_obj = curriculum_obj
    self._max_seq_length = max_len
    self._ops = ops
    self._token_by_char = token_by_char
    self._batch_size = batch_size

    # Construct the vocabulary.
    num_token_digits = 1 if token_by_char else curriculum_obj.max_length
    token_list = get_tokens(10 ** num_token_digits)
    self.vocab_size = 1
    for token in self.DEFAULT_START_TOKENS + token_list:
      if token not in self._vocab_dict:
        self._vocab_dict[token] = self.vocab_size
        self._inv_vocab_dict[self.vocab_size] = token
        self.vocab_size += 1

  @property
  def vocabulary(self):
    """List of strings, dataset vocabulary."""
    return self._vocab_dict.keys()

  def generate_flat_data(self):
    """Generates batched data in flat numpy arrays.

    Raises:
      ValueError: When too many generate calls are required.
    """
    # Construct the string statements.
    all_statements = []
    all_targets = []
    self.sequence_sizes_in = []
    self.sequence_sizes_out = []
    for _ in six.moves.range(self._batch_size):
      length, nest = self.curriculum_obj.fetch()
      seq_size_in = self._max_seq_length
      # Generate batch within max length.
      is_valid_sample = False
      tries_remaining = 10
      while not is_valid_sample:
        value, code = generate_code(length, nest, self._ops)
        tokens_in, seq_size_in = self.tokenize(
            code, self._max_seq_length, self._token_by_char)
        tokens_out, seq_size_out = self.tokenize(
            value, self._max_seq_length, self._token_by_char)
        is_valid_sample = self._max_seq_length >= seq_size_in
        if is_valid_sample:
          self.sequence_sizes_in.append(seq_size_in)
          self.sequence_sizes_out.append(seq_size_out)
        if tries_remaining == 0:
          raise ValueError("Could not generate a sample below the allowable "
                           "maximum, consider reducing either max_length or "
                           "max_nest.")
        else:
          tries_remaining -= 1
      all_statements += tokens_in
      all_targets += tokens_out
    # Store the flattened data.
    self.flat_data = np.array(all_statements, dtype=np.int64)
    self.num_tokens = self.flat_data.shape[0]
    self.flat_targets = np.array(all_targets, dtype=np.int64)
    self.num_tokens_target = self.flat_targets.shape[0]
    self.start_token = np.array(self.tokenize(
        [get_start_token()], 1)[0], dtype=np.int64)
    self.end_token = np.array(self.tokenize(
        [get_end_token()], 1)[0], dtype=np.int64)

  def tokenize(self, char_input, max_len, by_char=False):
    """Produces the list of integer indices corresponding to a token list.

    Args:
      char_input: The character string to be tokenized.
      max_len: Truncation length.
      by_char: If true each character is a token - otherwise alpha-numeric
               groupings are tokens.

    Returns:
      A padded list of string tokens and the true sequence length.

    Raises:
      ValueError: the token sequence is too long.
    """
    if by_char:
      tokenized_list = [self._vocab_dict[token] for token in char_input]
    else:
      tokenized_list = []
      compound_token = ""
      for token in char_input:
        # Compose alphanumeric inputs into compound tokens.
        add_number = compound_token.isdigit() and not token.isdigit()
        add_word = compound_token.isalpha() and not token.isalpha()
        if add_number or add_word:
          tokenized_list.append(self._vocab_dict[compound_token])
          compound_token = ""
        # Add token or build compound token.
        if token.isdigit():
          compound_token += token
        elif token.isalpha():
          compound_token += token
        else:
          tokenized_list.append(self._vocab_dict[token])
      if compound_token:
        tokenized_list.append(self._vocab_dict[compound_token])

    # To ensure uniform batch sequence length pad the sequence.
    seq_size = len(tokenized_list)
    if seq_size < max_len:
      padding = [self._vocab_dict[get_padding()]] * (max_len - seq_size)
      tokenized_list.extend(padding)
    elif seq_size > max_len:
      raise ValueError("Token sequence is too large: {}".format(
          len(tokenized_list)))

    return tokenized_list, seq_size

  def decode_to_string(self, token_list):
    """Produces a human-readable representation of the token list."""
    return "".join([self._inv_vocab_dict[token] for token in token_list])

  def decode_to_list(self, token_list):
    """Returns a list token index values."""
    return [self._inv_vocab_dict[token] for token in token_list]


# Task Types.
class TaskType(Enum):
  ALGEBRA = 1
  CONTROL = 2
  ALL = 3
  ALG_CTRL = 4
  ADDITION = 5
  COPY = 6
  DOUBLE = 7
  REVERSE = 8


# Task Groups.
class TaskGroups(Enum):
  PROG_TASKS = 1
  MEMORY_1 = 2
  MEMORY_2 = 3


class LearnToExecuteState(object):
  """Learn-To-Execute dataset state class.

  Generated sequences of constant time mini-programs.

  Modes:

  * `train`: Uses Combined Curriulum for training.
  * `test`: Uses Baseline Currculum for testing.
  """

  TASK_TYPE_OPS = {
      TaskType.ALGEBRA: [AddOp(), SubtractOp(), MultiplyOp(),],
      TaskType.CONTROL: [IfOp(), ForOp(),],
      TaskType.ALL: [AddOp(), SubtractOp(), MultiplyOp(), IfOp(), ForOp(),],
      TaskType.ALG_CTRL: [AddOp(), SubtractOp(), IfOp()],
      TaskType.ADDITION: [AddOp()],
      TaskType.COPY: [CopyOp()],
      TaskType.DOUBLE: [DoubleCopyOp()],
      TaskType.REVERSE: [ReverseOp()],
  }

  TASK_GROUPS = {
      TaskGroups.PROG_TASKS: (
          TaskType.ALGEBRA, TaskType.ALL, TaskType.ADDITION,
          TaskType.ALG_CTRL, TaskType.CONTROL),
      TaskGroups.MEMORY_1: (TaskType.COPY, TaskType.REVERSE),
      TaskGroups.MEMORY_2: (TaskType.DOUBLE,)
  }

  def __init__(self, batch_size, max_length, max_nesting, curriculum,
               token_by_char=True, task_type="alg-ctrl"):
    """Creates a LearnToExecute Dataset.

    Initializes the dataset task set and input annd target sequence shapes.
    Maximum sequence sizes for input and target are computed based on maximum
    possible assignments.  Also, curriculum is set, operations corresponding to
    the chosen task is set and the data source is initialized.

    Args:
      batch_size: (int). The number of elements in a mini-batch.
      max_length: (int). Maximum character length.
      max_nesting: (int). Maximum level of statement nesting.
      curriculum: (LTECurriculum). Curriculum strategy to use.
      token_by_char: (bool). Tokenize by character or words?
      task_type: (string) defines the task by allowable ops (see TASK_TYPE_OPS).

    Raises:
      ValueError: If task is invalid.
    """
    super(LearnToExecuteState, self).__init__()
    self._token_by_char = token_by_char
    # Compute the max number of steps possible to take.
    if task_type in self.TASK_GROUPS[TaskGroups.PROG_TASKS]:
      if token_by_char:
        outer_nests_term = (max_length * 3 + 10) * (max_nesting - 1)
        inner_nest_term = max_length * 4 + 10
        nest_tok_term = (max_nesting - 1) * 2
        self._num_steps_out = max_length * 2
      else:
        outer_nests_term = 10 * (max_nesting - 1)
        inner_nest_term = 11
        nest_tok_term = (max_nesting - 1) * 2
        self._num_steps_out = 1
      self._num_steps = outer_nests_term + inner_nest_term + nest_tok_term
    elif task_type in self.TASK_GROUPS[TaskGroups.MEMORY_1]:
      self._token_by_char = True
      self._num_steps = max_length + 1
      self._num_steps_out = max_length + 1
    elif task_type in self.TASK_GROUPS[TaskGroups.MEMORY_2]:
      self._token_by_char = True
      self._num_steps = max_length + 1
      self._num_steps_out = max_length * 2 + 1
    else:
      raise ValueError("Unknown task: {}.".format(task_type))

    self._batch_size = batch_size
    self._ops = LearnToExecuteState.get_task_ops(task_type)
    self._curriculum = curriculum
    num_steps = max(self._num_steps, self._num_steps_out)
    self._data_source = TokenDataSource(
        self._curriculum, self._batch_size, num_steps, self._ops,
        self._token_by_char)
    self.reset_data_source()

  @staticmethod
  def get_task_ops(task_type=TaskType.ALG_CTRL):
    """Returns an operations list based on the specified task index.

    Args:
      task_type: indicates the task type used.

    Returns:
      List of the eligible ops.
    """
    try:
      return LearnToExecuteState.TASK_TYPE_OPS[task_type]
    except KeyError:
      raise KeyError("Bad task_type '%s', check config." % task_type)

  @property
  def vocabulary(self):
    """List of strings, dataset vocabulary."""
    return self._data_source.vocabulary

  @property
  def vocab_size(self):
    return self._data_source.vocab_size

  def _np_one_hot(self, tensor, num_steps):
    tensor_oh = np.zeros((tensor.size, self.vocab_size))
    tensor_oh[np.arange(tensor.size), tensor.flat] = 1
    return tensor_oh.reshape(
        num_steps, self.batch_size, self.vocab_size).astype(np.float32)

  def reset_data_source(self):
    """Build the data source given the current curriculum state."""
    self._data_source.generate_flat_data()

  def evaluate_curriculum(self, loss):
    """If the currciulum state has updated rebuild the data source."""
    if self._curriculum.update(loss):
      self.reset_data_source()

  @property
  def num_steps(self):
    return self._num_steps

  @property
  def num_steps_out(self):
    return self._num_steps_out

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def curriculum(self):
    """Property returning curriculum object for this dataset."""
    return self._curriculum

  @property
  def level(self):
    return self._curriculum.current_level

  @property
  def seq_sizes_in(self):
    """Stores the input sequence size per batch."""
    return self._data_source.sequence_sizes_in[:self.batch_size]

  @property
  def seq_sizes_out(self):
    """Stores the target sequence size per batch."""
    return self._data_source.sequence_sizes_out[:self.batch_size]

  def make_batch(self):
    """Generator function for batchifying data for learning to execute.

    Yields:
      tuple:
        1. one-hot input tensor, representing programmatic input
        2. one-hot target tensor, the vealuation result.
        3. one-hot decoder target, start symbol added for sequence decoding.
        4. batch size tensor containing integer input sequence lengths.
        5. batch size tensor containing integer output sequence lengths.
    """
    while True:
      self.reset_data_source()
      obs = np.reshape(self._data_source.flat_data,
                       [self.batch_size, -1])[:, :self._num_steps].T
      target = np.reshape(
          self._data_source.flat_targets,
          [self.batch_size, -1])[:, :self._num_steps_out].T
      start_tokens = np.ndarray([1, self.batch_size], dtype=np.int32)
      start_tokens.fill(self._data_source.start_token[0])
      target_in = np.concatenate((start_tokens, target[:-1, :]), axis=0)
      yield (self._np_one_hot(obs, self._num_steps),
             self._np_one_hot(target, self._num_steps_out),
             self._np_one_hot(target_in, self._num_steps_out),
             self.seq_sizes_in,
             self.seq_sizes_out)

  def to_human_readable(self, data, label_batch_entries=True, indices=None,
                        sep="\n"):
    """Returns a human-readable version of a one-hot encoding of words.

    Args:
      data: (numpy.ndarray S x B x OH). One-hot encoding of words. S is
          sequence length, B is batch size, OH is one hot dimensionality.
      label_batch_entries: (bool). Whether to add numerical label before each
          batch element in the output string.
      indices: (list(int) or None). Used to select a subset of minibatch indices
          to print. None will print the whole minibatch.
      sep: (str) separator which separates the output for each batch. Defaults
          to the newline character.

    Returns:
      String composed from the data.
    """
    batch_size = data.shape[1]
    result = []
    indices = indices or six.moves.range(batch_size)
    for b in indices:
      index_seq = np.argmax(data[:, b], axis=1)
      prefix = "b_{}: ".format(b) if label_batch_entries else ""
      result.append(prefix + self._data_source.decode_to_string(index_seq))
    return sep.join(result)


# Sampling curriculum modes.
class Mode(Enum):
  TRAIN_COMBINE = 1
  TRAIN_MIX = 2
  TRAIN_NAIVE = 3
  TEST = 4


def LearnToExecute(   # pylint: disable=invalid-name
    batch_size, max_length=1, max_nesting=1, token_by_char=True,
    mode=Mode.TRAIN_COMBINE, loss_threshold=0.1,
    min_tries=DEFAULT_MIN_CURRICULUM_EVAL_TRIES, task_type=TaskType.ALG_CTRL):
  """Factory method for LearnToExecute Dataset module.

  Args:
    batch_size: (int). The number of elements in a mini-batch.
    max_length: (int). Maximum character length.
    max_nesting: (int). Maximum level of statement nesting.
    token_by_char: (bool). Tokenize by character or words?
    mode: (string). Either 'train', 'test'.
    loss_threshold: (int) curriculum threshold for error below which increase
        the task difficulty.
    min_tries: (int) minimum update tries for curriculum difficulty level.
    task_type: (string) defines the task by allowable ops (see TASK_TYPE_OPS).

  Returns:
    tf.Data.Dataset for LearnToExecute sample generator with the
    LearnToExecuteState monkey patched into the `state` attribute.

  Raises:
    ValueError: in case of bad `mode`.
  """
  # defaults mode to "train-combine"
  if mode == Mode.TRAIN_COMBINE:
    curriculum = CombineCurriculum(
        max_length, max_nesting, loss_threshold, min_tries=min_tries)
  elif mode == Mode.TRAIN_MIX:
    curriculum = MixCurriculum(
        max_length, max_nesting, loss_threshold, min_tries=min_tries)
  elif mode == Mode.TRAIN_NAIVE:
    curriculum = NaiveCurriculum(
        max_length, max_nesting, loss_threshold, min_tries=min_tries)
  elif mode == Mode.TEST:
    curriculum = BaselineCurriculum(
        max_length, max_nesting, loss_threshold, min_tries=0)
  else:
    raise ValueError("Invalid mode.")
  lte = LearnToExecuteState(batch_size, max_length, max_nesting,
                            curriculum, token_by_char, task_type=task_type)
  types_ = (tf.float32, tf.float32, tf.float32, tf.int64, tf.int64)
  shapes_ = (tf.TensorShape([lte.num_steps, batch_size, lte.vocab_size]),
             tf.TensorShape([lte.num_steps_out, batch_size, lte.vocab_size]),
             tf.TensorShape([lte.num_steps_out, batch_size, lte.vocab_size]),
             tf.TensorShape([batch_size,]),
             tf.TensorShape([batch_size,]))
  dataset = tf.data.Dataset.from_generator(lte.make_batch, types_, shapes_)
  dataset.state = lte
  return dataset
