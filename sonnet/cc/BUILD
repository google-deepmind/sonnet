licenses(["notice"])  # Apache 2.0 License

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

load(
    "//sonnet:tensorflow.bzl",
    "tf_gen_op_libs",
)

cc_library(
    name = "ops",
    deps = [
        ":resampler_op_lib",
    ],
    alwayslink = 1,
)

tf_gen_op_libs(
    op_lib_names = [
        "resampler",
    ],
)
