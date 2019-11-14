FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get -y install python3 cmake git make python3-pip libopenblas-dev python3-numpy python3-scipy python3-pandas
RUN python3 -c 'import numpy; import scipy; import pandas'
RUN pip3 install torch sh tqdm
RUN git clone https://github.com/BarclayII/dgl /opt/dgl
RUN cd /opt/dgl && git checkout pinsage-neighbors && git submodule update --init --recursive
RUN mkdir /opt/dgl/build && cd /opt/dgl/build && cmake -DUSE_CUDA=ON -DUSE_OPENMP=ON -DCUDA_ARCH_NAME=All .. && make -j4
RUN cd /opt/dgl/python && python3 setup.py install
RUN pip3 install sklearn
RUN python3 -c 'import dgl'
