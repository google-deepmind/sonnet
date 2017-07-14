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
# =============================================================================
load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda")

genrule(
    name = "setup_py",
    srcs = ["setup.py.tmpl"],
    outs = ["setup.py"],
    cmd = if_cuda("cat $< | sed 's/%%%PROJECT_NAME%%%/dm-sonnet-gpu/g' > $@",
                  "cat $< | sed 's/%%%PROJECT_NAME%%%/dm-sonnet/g' > $@")
)

sh_binary(
    name = "install",
    srcs = ["install.sh"],
    data = [
        "MANIFEST.in",
        ":setup_py",
        "//sonnet",
        "//sonnet/examples",
    ],
)
