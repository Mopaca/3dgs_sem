FROM 3dgs_sem_base:py38

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# 기본 유틸 설치
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    ca-certificates \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 현재 OS 확인용
RUN cat /etc/os-release

# deadsnakes PPA 추가 후 Python 3.10 설치
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3.10-distutils \
    && rm -rf /var/lib/apt/lists/*

# Python 3.10용 pip 설치
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# python / pip 별칭 추가
RUN ln -sf /usr/bin/python3.10 /usr/bin/python310 && \
    ln -sf /usr/local/bin/pip /usr/bin/pip310

# pip 업그레이드
RUN python3.10 -m pip install --upgrade pip setuptools wheel

# 버전 확인
RUN python3.10 --version && python3.10 -m pip --version

WORKDIR /workspace