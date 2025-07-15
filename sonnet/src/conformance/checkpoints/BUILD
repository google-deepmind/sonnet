load("//third_party/bazel_rules/rules_python/python:py_binary.bzl", "py_binary")

package(default_testonly = True)

licenses(["notice"])

py_binary(
    name = "generate",
    srcs = ["generate.py"],
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
