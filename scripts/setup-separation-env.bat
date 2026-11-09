@echo off
REM ===========================================================================
REM 为「重叠语音分离」创建独立环境（pyannote.audio 3.3.2）
REM
REM 为什么要独立环境：分离模型只支持 pyannote.audio 3.3.2，
REM 主服务跑 4.x，两者不能装在同一个 venv 里。
REM
REM 使用前还需要（一次性）：
REM   1. 用同一个 HF 账号登录后，分别打开并同意：
REM        https://hf.co/pyannote/separation-ami-1.0
REM        https://hf.co/pyannote/speech-separation-ami-1.0
REM   2. 项目根目录 .env 里配置 HF_TOKEN，并设置：
REM        SEPARATION_ENABLED=1
REM        SEPARATION_PYTHON=D:/projects/polyglot-media-summarizer/.venv-sep/Scripts/python.exe
REM ===========================================================================
setlocal
cd /d "%~dp0.."

if not exist ".venv-sep\Scripts\python.exe" (
    echo [1/4] 创建 .venv-sep ...
    python -m venv .venv-sep || goto :error
) else (
    echo [1/4] 复用已有 .venv-sep
)

echo [2/4] 升级 pip ...
".venv-sep\Scripts\python.exe" -m pip install --upgrade pip || goto :error

echo [3/4] 安装 torch / torchaudio（CUDA 12.4）...
REM 如果你的显卡驱动只支持 CUDA 12.1，把 cu124 改成 cu121
".venv-sep\Scripts\pip.exe" install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 || goto :error

echo [4/4] 安装 pyannote.audio[separation]==3.3.2 ...
".venv-sep\Scripts\pip.exe" install "pyannote.audio[separation]==3.3.2" || goto :error

echo.
echo 自检：
".venv-sep\Scripts\python.exe" -c "import torch, pyannote.audio as pa; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('pyannote', pa.__version__)"
if errorlevel 1 goto :error

echo.
echo 完成。把下面这行写进项目根目录的 .env：
echo SEPARATION_PYTHON=%CD%\.venv-sep\Scripts\python.exe
echo.
echo 首次分离会自动下载模型（约 1.3GB）。记得先同意两个 gated 模型的条款。
exit /b 0

:error
echo.
echo 安装失败。常见原因：
echo   - 没有 CUDA 12.x 驱动：把脚本里的 cu124 改成 cu121 或 cpu
echo   - pyannote 3.3.2 与 torch 版本冲突：先删掉 .venv-sep 再重试
echo   - 网络问题：配置 pip 镜像或代理
exit /b 1
