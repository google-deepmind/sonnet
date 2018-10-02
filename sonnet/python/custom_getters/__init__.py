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
"""Custom getters for use in TensorFlow and Sonnet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sonnet.python.custom_getters import bayes_by_backprop
from sonnet.python.custom_getters.context import Context
from sonnet.python.custom_getters.non_trainable import non_trainable
from sonnet.python.custom_getters.override_args import override_args
from sonnet.python.custom_getters.override_args import override_default_args
from sonnet.python.custom_getters.restore_initializer import restore_initializer
from sonnet.python.custom_getters.stop_gradient import stop_gradient
