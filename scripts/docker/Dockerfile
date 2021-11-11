FROM nvidia/cuda:10.1-devel-ubuntu18.04

RUN apt update
RUN apt install -y git cmake python3 wget
WORKDIR /root
RUN git clone https://github.com/aoli-al/HFuse
RUN git clone https://github.com/aoli-al/TorchKernel
RUN git clone https://github.com/aoli-al/ethminer

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda install -y pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch

WORKDIR /root
CMD sh
