FROM tensorflow/tensorflow:1.0.1

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list && \
    curl https://bazel.build/bazel-release.pub.gpg | apt-key add - 

# We need to add a custom PPA to pick up JDK8, since trusty doesn't
# have an openjdk8 backport.  openjdk-r is maintained by a reliable contributor:
# Matthias Klose (https://launchpad.net/~doko).  It will do until
# we either update the base image beyond 14.04 or openjdk-8 is
# finally backported to trusty; see e.g.
#   https://bugs.launchpad.net/trusty-backports/+bug/1368094
RUN add-apt-repository -y ppa:openjdk-r/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends openjdk-8-jdk openjdk-8-jre-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y bazel && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/local
RUN git clone --recursive https://github.com/deepmind/sonnet

WORKDIR /usr/local/sonnet/tensorflow

ENV CC_OPT_FLAGS=-march=native PYTHON_BIN_PATH=/usr/bin/python TF_NEED_MKL=0 TF_NEED_JEMALLOC=0 TF_NEED_GCP=0 TF_NEED_HDFS=0 TF_ENABLE_XLA=0 TF_NEED_OPENCL=0 TF_NEED_CUDA=0 GCC_HOST_COMPILER_PATH=/usb/bin/gcc

RUN ./configure

WORKDIR /usr/local/sonnet

RUN mkdir /tmp/sonnet && bazel build --config=opt :install && ./bazel-bin/install /tmp/sonnet && pip install /tmp/sonnet/*.whl

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python2.7/dist-packages/sonnet/python/ops

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

WORKDIR "/notebooks"

CMD ["/run_jupyter.sh", "--no-browser"]
