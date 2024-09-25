FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

RUN sed -i "s@http://archive.ubuntu.com@http://mirrors.aliyun.com@g" /etc/apt/sources.list && \
    sed -i "s@http://security.ubuntu.com@http://mirrors.aliyun.com@g" /etc/apt/sources.list && \
    sed -i "s@http://ports.ubuntu.com@http://mirrors.aliyun.com@g" /etc/apt/sources.list && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git vim watchman wget

RUN wget https://tzrec.oss-cn-beijing.aliyuncs.com/third_party/libidn11_1.33-2.2ubuntu2_amd64.deb && \
    apt-get install ./libidn11_1.33-2.2ubuntu2_amd64.deb

ADD pip.conf /root/.config/pip/pip.conf

RUN pip install fbgemm-gpu==0.8.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install torchmetrics==1.0.3 && \
    pip install torchrec==0.8.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install torch_tensorrt==2.4.0

ADD requirements.txt /root/requirements.txt
ADD requirements /root/requirements
RUN cd /root && pip install -r requirements.txt
