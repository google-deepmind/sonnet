load("//sonnet/src:build_defs.bzl", "snt_py_library", "snt_py_test")

package(default_visibility = ["//sonnet:__subpackages__", "//docs/ext:__subpackages__", "//examples:__subpackages__"])

licenses(["notice"])  # Apache 2.0 License

snt_py_library(
    name = "batch_norm",
    srcs = ["batch_norm.py"],
    deps = [
        "//sonnet/src:base",
        "//sonnet/src:batch_norm",
        "//sonnet/src:initializers",
        "//sonnet/src:metrics",
        "//sonnet/src:moving_averages",
        "//sonnet/src:once",
        "//sonnet/src:types",
        "//sonnet/src:utils",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "batch_norm_test",
    srcs = ["batch_norm_test.py"],
    deps = [
        ":batch_norm",
        ":replicator",
        # pip: absl/testing:parameterized
        "//sonnet/src:test_utils",
        # pip: tensorflow
    ],
)

snt_py_library(
    name = "replicator",
    srcs = ["replicator.py"],
    deps = [
        # pip: absl/logging
        # pip: contextlib2
        "//sonnet/src:initializers",
        # pip: tensorflow
    ],
)

snt_py_test(
    name = "replicator_test",
    srcs = ["replicator_test.py"],
    deps = [
        ":replicator",
        ":replicator_test_utils",
        "//sonnet/src:initializers",
        "//sonnet/src:test_utils",
        # pip: tensorflow
    ],
)

snt_py_library(
    name = "replicator_test_utils",
    testonly = 1,
    srcs = ["replicator_test_utils.py"],
    deps = [
        ":replicator",
        # pip: absl/logging
        # pip: tensorflow
        # https://github.com/google/pytype/blob/master/2.7_patches/
    ],
)
