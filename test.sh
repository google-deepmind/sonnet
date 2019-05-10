#!/bin/bash

# Pip installs the relevant dependencies and runs the Sonnet tests on CPU

set -e
set -x

N_JOBS=$(grep -c ^processor /proc/cpuinfo)

echo ""
echo "Bazel will use ${N_JOBS} concurrent job(s)."
echo ""

# Python dependencies.
pip install -r requirements.txt
pip install -r requirements-test.txt

# Install tensorflow.
pip install --upgrade tf-nightly-2.0-preview

# Run bazel test command. Double test timeouts to avoid flakes.
bazel test --jobs=${N_JOBS} --test_timeout 300,450,1200,3600 \
    --build_tests_only --test_output=errors \
    --cache_test_results=no \
    -- //sonnet/...
