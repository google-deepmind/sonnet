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

"""Module for experimental sonnet functions and classes.

This file contains functions and classes that are being tested until they're
either removed or promoted into the wider sonnet library.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from sonnet.python.modules import util

from tensorflow.python.util import deprecation


@deprecation.deprecated(
    "2017-08-01",
    "The @snt.experimental.reuse_vars decorator has been moved to "
    "@snt.reuse_variables. Please change to use the new location. ")
def reuse_vars(method):
  return util.reuse_variables(method)
