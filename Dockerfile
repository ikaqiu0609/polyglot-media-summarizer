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
# 2) code-server + 预装扩展
#    镜像内自带 code-server 时，CNB 云原生开发会以「单容器模式」启动（更省资源、启动更快）。
#
#    ⚠️ 单容器模式的代价：CNB 会直接用镜像里这个 code-server，不再额外拉起
#    cnbcool/default-dev-env 容器，而 CodeBuddy（tencent-cloud.coding-copilot）、
#    zsh 等默认工具都装在那个容器镜像里 —— 不在这里显式安装，打开 IDE 就会发现
#    「AI 助手不见了，得去插件市场重装」。所以下面必须把要用的扩展装齐。
#
#    扩展用 `|| echo` 兜底：Open VSX 偶发抽风时不至于让整个环境构建失败。
# ---------------------------------------------------------------------------
RUN curl -fsSL https://code-server.dev/install.sh | sh \
    && for ext in \
         cnbcool.cnb-welcome \
         tencent-cloud.coding-copilot \
         ms-python.python \
       ; do \
         code-server --install-extension "$ext" \
           || echo "[warn] 扩展安装失败（不阻断构建）: $ext"; \
       done \
    && code-server --list-extensions

# ---------------------------------------------------------------------------
# 3) Python 依赖
#    注意：requirements.txt 里不要写 torch，交给基础镜像提供；
#    murmurai-core 会顺带安装 pyannote-audio(>=4.x)、faster-whisper、ctranslate2、torchcodec 等。
#    pip 会额外装一个 nvidia-cudnn-cu12（>700MB），与镜像自带 cuDNN9 同大版本，可共存。
#
#    ⚠️ CNB 云原生开发的构建上下文不是仓库根目录，而是 .cnb.yml 里
#    docker.build.by 声明的那几个文件（外加 Dockerfile 自己）。
#    所以这里 COPY 的 requirements.txt 必须在 by 里列出来，
#    否则构建会直接失败：
#      COPY requirements.txt /tmp/requirements.txt
#      ERROR: failed to calculate checksum ...: "/requirements.txt": not found
# ---------------------------------------------------------------------------
WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"

# ---------------------------------------------------------------------------
# 4) 运行时目录
#    业务代码**不** COPY 进镜像：CNB 云原生开发把仓库直接挂载在 /workspace，
#    镜像里再 COPY 一份既多余、又需要把整仓写进 build.by。
#    想拿这个镜像独立跑（docker run），把仓库挂到 /workspace 即可，见 README A9。
# ---------------------------------------------------------------------------
RUN mkdir -p /workspace/data/uploads /workspace/data/audio /workspace/data/outputs \
    && chmod -R 777 /workspace/data

EXPOSE 8000

# 云原生开发环境里通常手动启动 uvicorn；容器直跑时用下面这条
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
