load("//third_party/bazel_rules/rules_python/python:py_binary.bzl", "py_binary")

package(
    default_testonly = True,
    default_visibility = ["//sonnet:__subpackages__", "//docs/ext:__subpackages__", "//examples:__subpackages__"],
)

licenses(["notice"])

py_binary(
    name = "generate",
    srcs = ["generate.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        # pip: absl:app
        # pip: absl/flags
        # pip: absl/logging
        "//sonnet/src/conformance:goldens",
        # pip: tensorflow
    ],
)

filegroup(
    name = "checkpoints",
    srcs = glob(["**/*"]),
)
