#!/bin/bash

# Pip installs the relevant dependencies and runs the Sonnet tests on CPU

set -e
set -x

N_JOBS=$(grep -c ^processor /proc/cpuinfo)

echo ""
echo "Bazel will use ${N_JOBS} concurrent job(s)."
echo ""

# Python dependencies.
# TODO(tamaranorman) Add requirements.txt and use here.
pip install mock absl-py six numpy wrapt

# Install tensorflow.
pip install --upgrade tf-nightly-2.0-preview

# Run bazel test command. Double test timeouts to avoid flakes.
bazel test --jobs=${N_JOBS} --test_timeout 300,450,1200,3600
    --build_tests_only --test_output=errors -- \
    //sonnet/...
