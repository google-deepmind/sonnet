licenses(["notice"])  # Apache 2.0 License

package(default_visibility = ["//visibility:public"])

exports_files(["LICENSE"])

load("//sonnet/examples/data:build_defs.bzl", "bzip2_decompress")

bzip2_decompress(
    name = "test",
    out = "ts.test.txt",
)

bzip2_decompress(
    name = "train",
    out = "ts.train.txt",
)

bzip2_decompress(
    name = "valid",
    out = "ts.valid.txt",
)
