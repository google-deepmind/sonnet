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

"""Removes the ":0" suffix from names in a checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf


tf.app.flags.DEFINE_string("source", None, "Source checkpoint")
tf.app.flags.DEFINE_string("target", None, "Target checkpoint")
tf.app.flags.DEFINE_boolean("dry_run", False, "Whether to do a dry run")

FLAGS = tf.app.flags.FLAGS


def _build_migrated_variables(checkpoint_reader, name_value_fn):
  """Builds the TensorFlow variables of the migrated checkpoint.

  Args:
    checkpoint_reader: A `tf.train.NewCheckPointReader` of the checkpoint to
      be read from.
    name_value_fn: Function taking two arguments, `name` and `value`, which
      returns the pair of new name and value for that a variable of that name.

  Returns:
    Tuple of a dictionary with new variable names as keys and `tf.Variable`s as
    values, and a dictionary that maps the old variable names to the new
    variable names.
  """

  names_to_shapes = checkpoint_reader.get_variable_to_shape_map()

  new_name_to_variable = {}
  name_to_new_name = {}

  for name in names_to_shapes:
    value = checkpoint_reader.get_tensor(name)
    new_name, new_value = name_value_fn(name, value)
    if new_name is None:
      continue

    name_to_new_name[name] = new_name
    new_name_to_variable[new_name] = tf.Variable(new_value)

  return new_name_to_variable, name_to_new_name


def remove_colon_zero(name):
  return name[:-2] if name.endswith(":0") else name


def main(unused_args):
  with tf.Graph().as_default():
    reader = tf.train.NewCheckpointReader(FLAGS.source)
    name_value_fn = lambda name, value: (remove_colon_zero(name), value)
    variables, name_to_new_name = _build_migrated_variables(
        reader, name_value_fn=name_value_fn)

    if not FLAGS.dry_run:
      init = tf.global_variables_initializer()
      saver = tf.train.Saver(variables)

      with tf.Session() as sess:
        sess.run(init)
        saver.save(sess, FLAGS.target)

  return name_to_new_name


if __name__ == "__main__":
  tf.app.run()
