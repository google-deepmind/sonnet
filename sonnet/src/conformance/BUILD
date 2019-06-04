package(
    default_testonly = True,
    default_visibility = ["//sonnet:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0 License

exports_files(["LICENSE"])

load("//sonnet/src:build_defs.bzl", "snt_py_library", "snt_py_test")

snt_py_library(
    name = "goldens",
    srcs = ["goldens.py"],
    deps = [
        # numpy
        # six
        "//sonnet",
        # tensorflow
    ],
)

snt_py_test(
    name = "api_test",
    srcs = ["api_test.py"],
    deps = [
        "//sonnet",
        "//sonnet/src:test_utils",
        # tensorflow
    ],
)

snt_py_test(
    name = "checkpoint_test",
    srcs = ["checkpoint_test.py"],
    data = [":checkpoints"],
    deps = [
        ":goldens",
        # absl/logging
        # absl/testing:absltest
        # absl/testing:parameterized
        "//sonnet",
        "//sonnet/src:replicator",
        "//sonnet/src:test_utils",
        # tensorflow
    ],
)

snt_py_test(
    name = "doctest_test",
    srcs = ["doctest_test.py"],
    deps = [
        # absl/testing:parameterized
        "//sonnet",
        "//sonnet/src:test_utils",
        # tensorflow
    ],
)

snt_py_test(
    name = "goldens_test",
    srcs = ["goldens_test.py"],
    deps = [
        ":goldens",
        "//sonnet",
        "//sonnet/src:test_utils",
        # tensorflow
    ],
)

snt_py_test(
    name = "pickle_test",
    srcs = ["pickle_test.py"],
    deps = [
        ":goldens",
        # absl/testing:parameterized
        "//sonnet/src:test_utils",
        # tensorflow
    ],
)

snt_py_test(
    name = "saved_model_test",
    srcs = ["saved_model_test.py"],
    deps = [
        ":goldens",
        # absl/testing:absltest
        # absl/testing:parameterized
        "//sonnet",
        "//sonnet/src:test_utils",
        # tensorflow
    ],
)
