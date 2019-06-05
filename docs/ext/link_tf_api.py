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

"""Reference TensorFlow API symbols.

This extension allows to reference TensorFlow API symbols using the
``:tf:`` role. For example, the following::

    Sonnet :py:`~base.Module` is based on :tf:`Module`.

generates a link to ``tf.Module``.
"""

# from __future__ import google_type_annotations

import urllib

from docutils import nodes
from docutils.parsers.rst import states
import tensorflow as tf
from typing import Any
from typing import List
from typing import Text
from typing import Tuple

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.util import tf_export

__version__ = "0.1"


# TODO(slebedev): make the version configurable or infer from ``tf``?
TF_VERSION = "2.0"
TF_API_BASE_URL = (
    "https://www.tensorflow.org/versions/r%s/api_docs/python/tf/" % TF_VERSION)


def tf_role_fn(
    typ: Text,
    rawtext: Text,
    text: Text,
    lineno: int,
    inliner: states.Inliner,
    options: Any = None,
    content: Any = None) -> Tuple[List[nodes.Node], List[nodes.system_message]]:
  """Generates a reference to a given TensorFlow API symbol.

  Only exported API symbols can be referenced. For example, non-exported
  :tf:`float32` will not produce a reference and will be rendered as
  plain-text.

  Args:
    typ: Type of the role. Fixed to ``"tf"``.
    rawtext: Raw contents of the role, e.g. ``":tf:`Module``"`.
    text: The `contents` of the role e.g. ``"Module"``.
    lineno: Line number of the parsed role.
    inliner: Inline reST markup parser. Used for error reporting.
    options: Unused.
    content: Unused.

  Returns:
    Generated reST nodes and system messages.
  """
  del options, content  # Unused.

  try:
    symbol = tf
    for chunk in text.split("."):
      symbol = getattr(tf, chunk)
  except AttributeError:
    canonical_name = ""
  else:
    canonical_name = tf_export.get_canonical_name_for_symbol(
        symbol,
        add_prefix_to_v1_names=True)

  xref = nodes.literal(rawtext, typ + "." + text, classes=["xref"])
  if not canonical_name:
    warning = (
        "unable to expand :%s:`%s`; symbol is not exported by TensorFlow."
        % (typ, text))
    inliner.reporter.warning(warning, line=lineno)
    return [xref], []
  else:
    canonical_url = urllib.parse.urljoin(
        TF_API_BASE_URL,
        canonical_name.replace(".", "/"))
    node = nodes.reference(
        rawtext,
        "",
        xref,
        internal=False,
        refuri=canonical_url)
    return [node], []


def setup(app):
  app.add_role("tf", tf_role_fn)

  return {
      "version": __version__,
      "parallel_read_safe": True,
      "parallel_write_safe": True,
  }
