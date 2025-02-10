ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

RUN sed -i "s@http://archive.ubuntu.com@http://mirrors.aliyun.com@g" /etc/apt/sources.list && \
    sed -i "s@http://security.ubuntu.com@http://mirrors.aliyun.com@g" /etc/apt/sources.list && \
    sed -i "s@http://ports.ubuntu.com@http://mirrors.aliyun.com@g" /etc/apt/sources.list && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        build-essential ca-certificates \
        ccache cmake gcc git vim watchman wget curl && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://tzrec.oss-cn-beijing.aliyuncs.com/third_party/libidn11_1.33-2.2ubuntu2_amd64.deb && \
    apt-get install ./libidn11_1.33-2.2ubuntu2_amd64.deb && rm libidn11_1.33-2.2ubuntu2_amd64.deb

ADD pip.conf /root/.config/pip/pip.conf
RUN curl -fsSL -v -o ~/miniconda.sh -O  "https://tzrec.oss-cn-beijing.aliyuncs.com/third_party/Miniforge3-Linux-x86_64.sh" && \
    chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda update -y -n base -c defaults conda && \
    /opt/conda/bin/conda install -y python=3.11 && \
    /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

ARG DEVICE
RUN case ${DEVICE} in \
        "cu124") pip install torch==2.6.0 fbgemm-gpu==1.1.0 --index-url https://download.pytorch.org/whl/cu124 && \
                 pip uninstall -y nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nvjitlink-cu12 nvidia-nvtx-cu12 && \
                 sed -i '/^Requires-Dist: nvidia-/d' /opt/conda/lib/python3.11/site-packages/torch-2.6.0+cu124.dist-info/METADATA && \
                 pip install torchmetrics==1.0.3 torch_tensorrt==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu124 && \
                 pip install torchrec==1.1.0 --index-url https://download.pytorch.org/whl/cu124 ;; \
        * )      pip install torch==2.6.0 fbgemm-gpu==1.1.0 --index-url https://download.pytorch.org/whl/cpu && \
                 pip install torchmetrics==1.0.3 && \
                 pip install torchrec==1.1.0 --index-url https://download.pytorch.org/whl/cpu ;; \
    esac && \
    /opt/conda/bin/conda clean -ya

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH

ADD requirements.txt /root/requirements.txt
ADD requirements /root/requirements
RUN cd /root && pip install -r requirements.txt && rm requirements.txt

RUN mkdir -p /home/pai/bin && \
    wget -O /home/pai/bin/prepare_dlc_environment https://tzrec.oss-cn-beijing.aliyuncs.com/third_party/pai/prepare_dlc_environment.sh && \
    chmod +x /home/pai/bin/prepare_dlc_environment && \
    apt-get update && apt-get install -y --no-install-recommends openssh-server && rm -rf /var/lib/apt/lists/* && service ssh start
ENV PATH $PATH:/home/pai/bin
ENV SHELL=/bin/bash
