@echo off
REM ===========================================================================
REM  Create an isolated environment for overlap speech separation.
REM
REM  Why a separate venv:
REM    pyannote/speech-separation-ami-1.0 requires pyannote.audio==3.3.2,
REM    while the main service runs pyannote.audio 4.x. They cannot coexist.
REM
REM  One-time prerequisites (use the same HuggingFace account):
REM    1. Accept the terms on BOTH pages:
REM         https://hf.co/pyannote/separation-ami-1.0
REM         https://hf.co/pyannote/speech-separation-ami-1.0
REM    2. Put HF_TOKEN in the project .env and set:
REM         SEPARATION_ENABLED=1
REM         SEPARATION_PYTHON=D:/projects/polyglot-media-summarizer/.venv-sep/Scripts/python.exe
REM ===========================================================================
setlocal
cd /d "%~dp0.."

set "PYTHON_EXE=.venv-sep\Scripts\python.exe"
set "PIP_EXE=.venv-sep\Scripts\pip.exe"

if not exist "%PYTHON_EXE%" (
    echo [1/4] Creating .venv-sep ...
    python -m venv .venv-sep || goto :error
) else (
    echo [1/4] Reusing existing .venv-sep
)

echo [2/4] Upgrading pip ...
"%PYTHON_EXE%" -m pip install --upgrade pip || goto :error

echo [3/4] Installing torch / torchaudio (CUDA 12.4) ...
REM Python 3.13 + Windows only has torch 2.6.0+cu124 in this index.
REM If your Python is 3.10-3.12 you may pin 2.5.1 instead.
"%PIP_EXE%" install "torch==2.6.0+cu124" "torchaudio==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124 || goto :error

echo [4/4] Installing pyannote.audio[separation]==3.3.2 ...
"%PIP_EXE%" install "pyannote.audio[separation]==3.3.2" || goto :error

echo.
echo Self-check:
"%PYTHON_EXE%" -c "import torch, torchaudio, pyannote.audio as pa; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('torchaudio', torchaudio.__version__); print('pyannote', pa.__version__)"
if errorlevel 1 goto :error

echo.
echo Done. Add this line to the project .env:
echo SEPARATION_PYTHON=%CD%\.venv-sep\Scripts\python.exe
echo.
echo The first separation run downloads about 1.3 GB of weights.
echo Make sure both gated models are accepted on HuggingFace.
exit /b 0

:error
echo.
echo Installation failed. Common causes:
echo   - No CUDA 12.x driver: replace cu124 with cu121 or cpu in this script
echo   - pyannote 3.3.2 / torch conflict: delete .venv-sep and retry
echo   - Network issue: configure a pip mirror or proxy
exit /b 1
