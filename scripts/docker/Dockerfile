FROM nvidia/cuda:10.1-devel-ubuntu18.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt update
RUN apt install -y git cmake python3 wget clang-10 libclang-10-dev zlibc zlib1g-dev libdbus-1-dev libboost-all-dev clang mesa-common-dev vim

ENV CC="clang"
ARG CC="clang"
ENV CXX="clang++"
ARG CXX="clang++"
ENV CUDA_HOME='/usr/local/cuda'
ARG CUDA_HOME='/usr/local/cuda'


WORKDIR /root
RUN git clone https://github.com/aoli-al/HFuse
WORKDIR /root/HFuse/deps/

WORKDIR /root/HFuse/
RUN mkdir build
WORKDIR /root/HFuse/build
RUN cmake ..
RUN make -j4


WORKDIR /root
RUN git clone https://github.com/ezyang/nvprof2json
RUN git clone https://github.com/aoli-al/TorchKernel
RUN git clone https://github.com/aoli-al/ethminer
RUN git clone --depth 1 https://github.com/aoli-al/pytorch
WORKDIR /root/ethminer
RUN git checkout ae
RUN git submodule update --init --recursive

WORKDIR /root
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py37_4.10.3-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py37_4.10.3-Linux-x86_64.sh 

RUN conda install -y pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch 
RUN pip3 install matplotlib

RUN mv /root/TorchKernel/Half.h /root/miniconda3/lib/python3.7/site-packages/torch/include/c10/util/Half.h

ENV CC="gcc"
ARG CC="gcc"
ENV CXX="g++"
ARG CXX="g++"

WORKDIR /root/TorchKernel
RUN python3 prepare_compiler_command.py
# RUN ./build.sh

WORKDIR /root/ethminer
# RUN mkdir build
# WORKDIR /root/ethminer/build
# RUN cmake ..
# RUN make fuser -j4

WORKDIR /root

RUN wget https://developer.download.nvidia.com/compute/cuda/11.5.0/local_installers/cuda_11.5.0_495.29.05_linux.run \
      && sh cuda_11.5.0_495.29.05_linux.run --toolkit --silent || true

RUN ln -sfn /usr/local/cuda-10.1/ /usr/local/cuda

#

CMD sh
