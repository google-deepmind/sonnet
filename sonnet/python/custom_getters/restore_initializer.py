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
"""Custom getter which uses snt.restore_initializer to restore all variables.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf


def restore_initializer(filename, name_fn=None,
                        collection=tf.GraphKeys.GLOBAL_VARIABLES):
  """Custom getter to restore all variables with `snt.restore_initializer`.

  Args:
    filename: The filename of the checkpoint.
    name_fn: A function which can map the name of the variable requested. This
      allows restoring variables with values having different names in the
      checkpoint.
    collection: Only set the restore initializer for variables in this
      collection. If `None`, it will attempt to restore all variables. By
      default `tf.GraphKeys.GLOBAL_VARIABLES`.

  Returns:
    A restore_initializer custom getter, which is a function taking arguments
    (getter, name, *args, **kwargs).
  """

  def _restore_initializer(getter, name, *args, **kwargs):
    """Gets variable with restore initializer."""

    # Work out what collections this variable will go in.
    collections = kwargs["collections"]
    if collections is None:
      collections = [tf.GraphKeys.GLOBAL_VARIABLES]

    if (kwargs["trainable"]
        and tf.GraphKeys.TRAINABLE_VARIABLES not in collections):
      collections += [tf.GraphKeys.TRAINABLE_VARIABLES]

    if collection is None or collection in collections:
      # We don't make use of the 'scope' argument for restore_initializer as we
      # might want to change the name in more complex ways, such as removing the
      # scope prefix as well.
      if name_fn is not None:
        var_name_in_checkpoint = name_fn(name)
      else:
        var_name_in_checkpoint = name

      tf.logging.info("Restoring '%s' from '%s' into variable '%s'",
                      var_name_in_checkpoint,
                      filename,
                      name)

      kwargs["initializer"] = snt.restore_initializer(
          filename, var_name_in_checkpoint, scope="")

    return getter(name, *args, **kwargs)

  return _restore_initializer
