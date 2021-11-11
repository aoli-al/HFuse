FROM nvidia/cuda:10.1-devel-ubuntu18.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt update
RUN apt install -y git cmake python3 wget
WORKDIR /root
RUN git clone https://github.com/aoli-al/HFuse
WORKDIR /root/HFuse/deps/
RUN git clone --depth 1 https://github.com/aoli-al/llvm-project 
RUN mkdir llvm-project/build
WORKDIR /root/HFuse/deps/llvm-project/build
RUN cmake -DLLVM_ENABLE_PROJECTS="clang"  -DCMAKE_INSTALL_PREFIX=/root/HFuse/deps/llvm-project/build -DLLVM_ENABLE_RTTI=ON -DCMAKE_BUILD_TYPE=Release ../llvm
RUN make -j6 install

WORKDIR /root/HFuse/
RUN mkdir build

WORKDIR /root
RUN git clone https://github.com/aoli-al/TorchKernel
RUN git clone https://github.com/aoli-al/ethminer

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-py37_4.10.3-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py37_4.10.3-Linux-x86_64.sh 


RUN conda install -y pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch

WORKDIR /root
CMD sh
