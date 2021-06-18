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
"""Sonnet mixed precision built for TensorFlow 2."""

from sonnet.src.mixed_precision import disable
from sonnet.src.mixed_precision import enable
from sonnet.src.mixed_precision import modes
from sonnet.src.mixed_precision import scope

__all__ = (
    "disable",
    "enable",
    "modes",
    "scope",
)
