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

import functools
from typing import Any, List, Tuple
from urllib import parse as urlparse

from docutils import nodes
from docutils.parsers.rst import states
import tensorflow as tf

from tensorflow.python.util import tf_export  # pylint: disable=g-direct-tensorflow-import


__version__ = "0.1"

# TODO(slebedev): make the version configurable or infer from ``tf``?
TF_VERSION = "2.0"
TF_API_BASE_URL = (
    "https://www.tensorflow.org/versions/r%s/api_docs/python/tf/" % TF_VERSION)


def tf_role_fn(
    typ: str,
    rawtext: str,
    text: str,
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

  canonical_url = tf_doc_url(text)
  xref = nodes.literal(rawtext, typ + "." + text, classes=["xref"])
  if not canonical_url:
    warning = (
        "unable to expand :%s:`%s`; symbol is not exported by TensorFlow." %
        (typ, text))
    inliner.reporter.warning(warning, line=lineno)
    return [xref], []
  else:
    node = nodes.reference(
        rawtext, "", xref, internal=False, refuri=canonical_url)
    return [node], []


def tf_doc_url(text):
  """Retrieves the TensorFlow doc URL for the given symbol.

  Args:
    text: A string for a symbol inside TF (e.g. ``"optimizers.Adam"``).

  Returns:
    A string URL linking to the TensorFlow doc site or ``None`` if a URL could
    not be resolved.
  """
  get_tf_name = functools.partial(
      tf_export.get_canonical_name_for_symbol, add_prefix_to_v1_names=True)

  try:
    prev_symbol = None
    symbol = tf
    for chunk in text.split("."):
      prev_symbol = symbol
      symbol = getattr(prev_symbol, chunk)
  except AttributeError:
    return None

  canonical_name = get_tf_name(symbol)

  # Check if we're looking at a method reference (e.g. "TensorArray.read").
  if prev_symbol and not canonical_name:
    prev_canonical_name = get_tf_name(prev_symbol)
    if prev_canonical_name:
      canonical_name = prev_canonical_name + "#" + text.split(".")[-1]

  if not canonical_name:
    return None

  return urlparse.urljoin(TF_API_BASE_URL, canonical_name.replace(".", "/"))


def setup(app):
  app.add_role("tf", tf_role_fn)

  return {
      "version": __version__,
      "parallel_read_safe": True,
      "parallel_write_safe": True,
  }
