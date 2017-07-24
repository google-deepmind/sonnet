#!/usr/bin/env bash
# Copyright 2017 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

set -e

PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"

function main() {
  if [ $# -lt 1 ] ; then
    echo "No destination dir provided"
    exit 1
  fi

  # Create the directory, then do dirname on a non-existent file inside it to
  # give us an absolute paths with tilde characters resolved to the destination
  # directory. Readlink -f is a cleaner way of doing this but is not available
  # on a fresh macOS install.
  mkdir -p "$1"
  DEST="$(dirname $1/does_not_exist)"
  echo "=== destination directory: ${DEST}"

  TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)

  echo $(date) : "=== Using tmpdir: ${TMPDIR}"

  cp -LR bazel-bin/install.runfiles/sonnet/* "${TMPDIR}"

  pushd ${TMPDIR}
  echo $(date) : "=== Building wheel"

  if [[ "$2" == "" ]] ; then
    python setup.py bdist_wheel >/dev/null
  else
    # Use custom location of python specified in argument:
    "$2" setup.py bdist_wheel >/dev/null
  fi

  cp dist/* ${DEST}
  popd
  rm -rf ${TMPDIR}
  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

main "$@"
