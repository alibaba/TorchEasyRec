ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN sed -i "s@http://archive.ubuntu.com@http://mirrors.aliyun.com@g" /etc/apt/sources.list.d/ubuntu.sources && \
    sed -i "s@http://security.ubuntu.com@http://mirrors.aliyun.com@g" /etc/apt/sources.list.d/ubuntu.sources && \
    sed -i "s@http://ports.ubuntu.com@http://mirrors.aliyun.com@g" /etc/apt/sources.list.d/ubuntu.sources && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        build-essential ca-certificates \
        ccache cmake gcc git vim watchman wget curl sudo iproute2 && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://tzrec.oss-cn-beijing.aliyuncs.com/third_party/libidn11_1.33-2.2ubuntu2_amd64.deb && \
    apt-get install ./libidn11_1.33-2.2ubuntu2_amd64.deb && rm libidn11_1.33-2.2ubuntu2_amd64.deb

ADD pip.conf /root/.config/pip/pip.conf

RUN pip install torch==2.7.0 -i https://art-pub.eng.t-head.cn/artifactory/api/pypi/ptgai-pypi_ppu_ubuntu_cu128_index/simple/ && \
    pip install fbgemm-gpu==1.2.0 faiss==1.11.0 -i https://art-pub.eng.t-head.cn/artifactory/api/pypi/ptgai-pypi_ppu_ubuntu_cu128_index/simple/ && \
    pip install torchmetrics==1.0.3 tensordict && \
    pip install torchrec==1.2.0

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ADD requirements /root/requirements
ADD requirements.txt /root/requirements.txt
ADD requirements-ppu.txt /root/requirements-ppu.txt
RUN cd /root && pip install -r requirements-ppu.txt && \
    rm -rf requirements requirements.txt requirements-ppu.txt

RUN mkdir -p /home/pai/bin && \
    wget -O /home/pai/bin/prepare_dlc_environment https://tzrec.oss-cn-beijing.aliyuncs.com/third_party/pai/prepare_dlc_environment.sh && \
    chmod +x /home/pai/bin/prepare_dlc_environment && \
    apt-get update && apt-get install -y --no-install-recommends openssh-server && rm -rf /var/lib/apt/lists/* && service ssh start
ENV PATH $PATH:/home/pai/bin

ENV SHELL=/bin/bash
RUN echo "dash dash/sh boolean false" | debconf-set-selections \
    && DEBIAN_FRONTEND=noninteractive dpkg-reconfigure dash
