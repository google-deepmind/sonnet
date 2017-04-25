# Description:
#   Custom ops for use in sonnet.

licenses(["notice"])  # Apache 2.0 License

exports_files(["LICENSE"])

package(default_visibility = ["//visibility:public"])

load(
    "//sonnet:tensorflow.bzl",
    "tf_kernel_library",
)

cc_library(
    name = "all_kernels",
    deps = [
        ":resampler_op",
    ],
)

tf_kernel_library(
    name = "resampler_op",
    prefix = "resampler_op",
    deps = [
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//third_party/eigen3",
        "@protobuf//:protobuf",
    ],
    alwayslink = 1,
)
