# syntax=docker/dockerfile:1.7
#
# 基础镜像选型说明：
#   官方 torch 镜像已内置 torch 2.9.0 + cu128 + torchaudio + cuDNN9，
#   而 murmurai-core 1.0.6 硬性要求 torch>=2.9.0 / torchaudio>=2.9.0 / nvidia-cudnn-cu12>=9.1.0，
#   直接复用可避免二次安装 CUDA 版 torch（省数 GB 且不会踩 torch 版本冲突）。
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

# 非交互式 apt
ENV DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# 模型缓存统一放 /workspace/.cache，云原生开发环境下可随工作区保留（已在 .gitignore 中排除）
ENV HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch

# 长音频推理时减少显存碎片导致的 OOM（PyTorch 2.x 支持）
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 业务默认值，可在 .cnb.yml 或运行时覆盖
ENV ASR_MODEL=large-v3 \
    ASR_COMPUTE_TYPE=float16 \
    ASR_BATCH_SIZE=16

# ---------------------------------------------------------------------------
# 1) 系统依赖
#    ffmpeg        : 音频抽取 / 按时长切片 / 转码
#    libsndfile1   : soundfile 读写 WAV（whisperx/murmurai 依赖）
#    openssh-server: 支持 VSCode/Cursor Remote-SSH 接入云原生开发环境
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        git \
        curl \
        ca-certificates \
        wget \
        unzip \
        procps \
        openssh-server \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2) code-server
#    镜像内自带 code-server 时，CNB 云原生开发会以「单容器模式」启动（更省资源、启动更快）。
#    若不需要网页版 IDE，可整段删除，届时会退化为双容器模式。
# ---------------------------------------------------------------------------
RUN curl -fsSL https://code-server.dev/install.sh | sh \
    && code-server --install-extension cnbcool.cnb-welcome \
    && code-server --install-extension ms-python.python \
    && echo "code-server installed"

# ---------------------------------------------------------------------------
# 3) Python 依赖
#    注意：requirements.txt 里不要写 torch，交给基础镜像提供；
#    murmurai-core 会顺带安装 pyannote-audio(>=4.x)、faster-whisper、ctranslate2、torchcodec 等。
#    pip 会额外装一个 nvidia-cudnn-cu12（>700MB），与镜像自带 cuDNN9 同大版本，可共存。
# ---------------------------------------------------------------------------
WORKDIR /workspace
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"

# ---------------------------------------------------------------------------
# 4) 业务代码（放在依赖之后，改代码不会导致重装依赖）
# ---------------------------------------------------------------------------
COPY . /workspace
RUN mkdir -p /workspace/data/uploads /workspace/data/audio /workspace/data/outputs \
    && chmod -R 777 /workspace/data

EXPOSE 8000

# 云原生开发环境里通常手动启动 uvicorn；容器直跑时用下面这条
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
