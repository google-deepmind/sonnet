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
"""Binary to generate golden checkpoint tests."""

import os
import re

from absl import app
from absl import flags
from absl import logging
from sonnet.src.conformance import goldens
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("golden_dir",
                    "sonnet/src/conformance/checkpoints/",
                    "Directory where golden files are to be found.")
flags.DEFINE_string("filter", ".*", "Filter to a specific golden by name.")
flags.DEFINE_bool("regenerate", False,
                  "Whether to regnerate existing checkpoints.")
flags.DEFINE_bool("dry_run", True, "Whether to actually apply changes.")


def safe_mkdir(directory):
  if FLAGS.dry_run:
    logging.warning("[DRY RUN] Would create %r", directory)
  else:
    logging.info("Creating %r", directory)
    os.mkdir(directory)


def safe_unlink(path):
  if FLAGS.dry_run:
    logging.warning("[DRY RUN] Would delete %r", path)
  else:
    logging.info("Deleting %r", path)
    os.unlink(path)


def main(unused_argv):
  del unused_argv

  for _, name, cls in goldens.list_goldens():
    if not re.match(FLAGS.filter, name):
      continue

    checkpoint_dir = os.path.join(FLAGS.golden_dir, name)
    exists = os.path.exists(checkpoint_dir)
    if exists and not FLAGS.regenerate:
      logging.info("Skipping %s since it exists and --regenerate=false", name)
      continue

    logging.info("Processing %s", name)
    if not exists:
      safe_mkdir(checkpoint_dir)
    else:
      # Clear out old files.
      for file_name in os.listdir(checkpoint_dir):
        safe_unlink(os.path.join(checkpoint_dir, file_name))

    # Create the module to checkpoint.
    golden = cls()
    module = golden.create_module()
    golden.create_all_variables(module)
    for var in module.variables:
      var.assign(goldens.range_like(var))

    # Create a checkpoint and save the values to it.
    checkpoint = tf.train.Checkpoint(module=module)
    if FLAGS.dry_run:
      logging.warning("[DRY RUN] Would save %r to %r", module, checkpoint_dir)
    else:
      file_prefix = os.path.join(checkpoint_dir, "checkpoint")
      logging.info("Saving to checkpoint %s.", file_prefix)
      checkpoint.save(file_prefix=file_prefix)


if __name__ == "__main__":
  app.run(main)
