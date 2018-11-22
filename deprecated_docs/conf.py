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
"""Sphinx configuration."""

import pypandoc
from recommonmark.transform import AutoStructify


def pandoc_convert(app, unused_a, unused_b, unused_c, unused_d, lines):
  if not lines:
    return
  input_format = app.config.pandoc_use_parser
  data = '\n'.join(lines).encode('utf-8')
  data = pypandoc.convert(data, 'rst', format=input_format)
  new_lines = data.split('\n')
  del lines[:]
  lines.extend(new_lines)


def setup(app):
  app.add_config_value('pandoc_use_parser', 'markdown', True)
  app.connect('autodoc-process-docstring', pandoc_convert)
  app.add_config_value('recommonmark_config', {
      'auto_to_tree_section': 'Contents'}, True)
  app.add_transform(AutoStructify)


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
]
templates_path = ['_templates']
source_suffix = ['.rst', '.md']
source_parsers = {
    '.md': 'recommonmark.parser.CommonMarkParser',
}
master_doc = 'index'
project = u'sonnet'
copyright = u'2017, Sonnet Authors'  # pylint: disable=redefined-builtin
version = 'git'
release = 'git'
exclude_patterns = ['_build']
pygments_style = 'sphinx'
html_theme = 'alabaster'
html_static_path = ['_static']
htmlhelp_basename = 'sonnetdoc'
latex_documents = [
    ('index', 'sonnet.tex', u'sonnet Documentation',
     u'Sonnet Authors', 'manual'),
]
man_pages = [
    ('index', 'sonnet', u'sonnet Documentation',
     [u'Sonnet Authors'], 1)
]
texinfo_documents = [
    ('index', 'sonnet', u'sonnet Documentation',
     u'Sonnet Authors', 'sonnet', 'One line description of project.',
     'Miscellaneous'),
]
