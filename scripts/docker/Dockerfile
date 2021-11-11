FROM nvidia/cuda:10.1-devel-ubuntu18.04

RUN apt update
RUN apt install -y git cmake
WORKDIR /root
RUN git clone https://github.com/aoli-al/HFuse
RUN git clone https://github.com/aoli-al/TorchKernel
RUN git clone https://github.com/aoli-al/ethminer

WORKDIR /root
CMD sh
