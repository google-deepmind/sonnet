"""Utilities for working with example data."""

def bzip2_decompress(name, out):
    native.genrule(
        name = name,
        srcs = [out + ".bz2"],
        outs = [out],
        cmd = "bzip2 -d -c $(SRCS) > $(OUTS)",
    )
