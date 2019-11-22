FROM nvidia/cuda:10.1-base-ubuntu18.04

RUN apt-get update --fix-missing && \
    apt-get install -y curl python3 python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

RUN pip install