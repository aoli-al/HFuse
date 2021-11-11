FROM nvidia/cuda:10.1-devel-ubuntu18.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
ENV CC="clang-10"
ARG CC="clang-10"
ENV CXX="clang++-10"
ARG CXX="clang++-10"


RUN apt update
RUN apt install -y git cmake python3 wget clang-10 libclang-10-dev zlibc zlib1g-dev
WORKDIR /root
RUN git clone https://github.com/aoli-al/HFuse
WORKDIR /root/HFuse/deps/

WORKDIR /root/HFuse/
RUN mkdir build
WORKDIR /root/HFuse/build
RUN cmake ..
RUN make -j4


WORKDIR /root
RUN git clone https://github.com/aoli-al/TorchKernel
RUN git clone https://github.com/aoli-al/ethminer
RUN git clone --depth 1 https://github.com/aoli-al/pytorch

WORKDIR /root/TorchKernel
RUN python3 prepare_compiler_command.py


RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py37_4.10.3-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py37_4.10.3-Linux-x86_64.sh 


RUN conda install -y pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch

CMD sh
