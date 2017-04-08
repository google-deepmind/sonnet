FROM tensorflow/tensorflow

RUN apt-get update && apt-get install -y git curl

RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && curl https://bazel.build/bazel-release.pub.gpg | apt-key add - 

RUN apt-get update && apt-get install -y bazel

WORKDIR /usr/local
# https://github.com/deepmind/sonnet/pull/6
# RUN git clone --recursive https://github.com/deepmind/sonnet
RUN git clone --recursive https://github.com/roman3017/sonnet.git

WORKDIR /usr/local/sonnet/tensorflow

ENV CC_OPT_FLAGS=-march=native PYTHON_BIN_PATH=/usr/bin/python TF_NEED_MKL=0 TF_NEED_JEMALLOC=0 TF_NEED_GCP=0 TF_NEED_HDFS=0 TF_ENABLE_XLA=0 TF_NEED_OPENCL=0 TF_NEED_CUDA=0 GCC_HOST_COMPILER_PATH=/usb/bin/gcc

RUN ./configure

WORKDIR /usr/local/sonnet

RUN mkdir /tmp/sonnet && bazel build --config=opt :install && ./bazel-bin/install /tmp/sonnet && pip install /tmp/sonnet/*.whl

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

WORKDIR "/notebooks"

CMD ["/run_jupyter.sh", "--no-browser"]
