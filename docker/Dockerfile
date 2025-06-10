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
        "cu126") wget https://tzrec.oss-cn-beijing.aliyuncs.com/third_party/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
                 dpkg -i cuda-keyring_1.1-1_all.deb && \
                 apt-get update && \
                 apt-get install cuda-compat-12-6 && \
                 rm -rf /var/lib/apt/lists/* ;; \
    esac
RUN case ${DEVICE} in \
        "cu126") pip install torch==2.7.0 fbgemm-gpu==1.2.0 --index-url https://download.pytorch.org/whl/cu126 && \
                 pip install torchmetrics==1.0.3 torch_tensorrt==2.7.0 --extra-index-url https://download.pytorch.org/whl/cu126 && \
                 pip install torchrec==1.2.0 --index-url https://download.pytorch.org/whl/cu126 ;; \
        * )      pip install torch==2.7.0 fbgemm-gpu==1.2.0 --index-url https://download.pytorch.org/whl/cpu && \
                 pip install torchmetrics==1.0.3 && \
                 pip install torchrec==1.2.0 --index-url https://download.pytorch.org/whl/cpu ;; \
    esac && \
    /opt/conda/bin/conda clean -ya

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ARG LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}

ADD requirements /root/requirements
ADD requirements.txt /root/requirements.txt
ADD requirements-cpu.txt /root/requirements-cpu.txt
ADD requirements-gpu.txt /root/requirements-gpu.txt
RUN cd /root && \
    case ${DEVICE} in \
        "cu126") pip install -r requirements-gpu.txt ;; \
        * )      pip install -r requirements-cpu.txt ;; \
    esac && \
    rm -rf requirements requirements.txt requirements-cpu.txt requirements-gpu.txt && \
    /opt/conda/bin/conda clean -ya

RUN mkdir -p /home/pai/bin && \
    wget -O /home/pai/bin/prepare_dlc_environment https://tzrec.oss-cn-beijing.aliyuncs.com/third_party/pai/prepare_dlc_environment.sh && \
    chmod +x /home/pai/bin/prepare_dlc_environment && \
    apt-get update && apt-get install -y --no-install-recommends openssh-server && rm -rf /var/lib/apt/lists/* && service ssh start
ENV PATH $PATH:/home/pai/bin
ENV SHELL=/bin/bash
