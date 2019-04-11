# Description:
#   Protocol buffers for the Sonnet library.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0 License

exports_files(["LICENSE"])

proto_library(
    name = "module_proto",
    srcs = ["module.proto"],
)

py_library(
    name = "module_pb2",
    srcs = [
        "__init__.py",
        ":gen_module_pb2",
    ],
)

genrule(
    name = "gen_module_pb2",
    srcs = ["module.proto"],
    outs = ["module_pb2.py"],
    cmd = "$(location @com_google_protobuf//:protoc) --python_out=$(@D) -I$$(dirname $<) $<",
    tools = ["@com_google_protobuf//:protoc"],
)
