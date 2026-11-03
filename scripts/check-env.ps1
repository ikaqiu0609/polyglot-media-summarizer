#requires -Version 5.1
<#
.SYNOPSIS
    Polyglot Media Summarizer —— 本地环境自检（Windows）。

.DESCRIPTION
    逐项检查 README「部署 B Windows 本地」里的关键前提：
      1. Python 版本（3.10 ~ 3.13）
      2. 虚拟环境 .venv
      3. Python 依赖（fastapi / openai / murmurai ...）
      4. torch 与 CUDA
      5. cuDNN 9（ctranslate2 能否看到显卡）
      6. FFmpeg / FFprobe 路径
      7. .env 与两个密钥
      8. 数据目录与 ASR 参数（8G 显存会提示降档）

    有任何 [FAIL] 都会以退出码 1 结束；[WARN] 不影响启动。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\check-env.ps1

.EXAMPLE
    .\scripts\check-env.ps1 -Python "D:\Python312\python.exe"
#>
[CmdletBinding()]
param(
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$env:PYTHONUTF8 = "1"

$script:fail = 0
$script:warn = 0

function Step($text) { Write-Host ""; Write-Host $text -ForegroundColor Cyan }
function Ok($text)   { Write-Host "  [OK]   $text" -ForegroundColor Green }
function Warn($text) { Write-Host "  [WARN] $text" -ForegroundColor Yellow; $script:warn++ }
function Fail($text) { Write-Host "  [FAIL] $text" -ForegroundColor Red;    $script:fail++ }
function Hint($text) { Write-Host "         -> $text" -ForegroundColor DarkGray }

Write-Host "==== Polyglot Media Summarizer · 环境自检 ====" -ForegroundColor White
Write-Host "项目目录: $ProjectRoot"

# ---------------------------------------------------------------------------
# 1. 定位 Python（优先 .venv）
# ---------------------------------------------------------------------------
Step "1/8 Python 解释器"

$venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$usingVenv = $false

if ($Python -and -not (Test-Path $Python)) {
    $resolved = Get-Command $Python -ErrorAction SilentlyContinue
    if ($resolved) { $Python = $resolved.Source }
}
if (-not $Python) {
    if (Test-Path $venvPython) {
        $Python = $venvPython
        $usingVenv = $true
    } else {
        $cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($cmd) { $Python = $cmd.Source }
    }
}
if (-not $Python -or -not (Test-Path $Python)) {
    Fail "找不到 Python"
    Hint "先完成 README B2：winget install --id Python.Python.3.12 -e"
    Hint "然后 B5：py -3.12 -m venv .venv"
    exit 1
}
if ($usingVenv) {
    Ok "使用虚拟环境: $Python"
} else {
    Warn "未找到 .venv，使用系统 Python: $Python"
    Hint "建议先执行 B5：py -3.12 -m venv .venv 并安装依赖"
}

# ---------------------------------------------------------------------------
# 2. 一次性探测（写成临时 .py 再执行，避免超长命令行与引号转义问题）
# ---------------------------------------------------------------------------
$probePath = Join-Path $env:TEMP ("pms_probe_{0}.py" -f $PID)
$probeCode = @'
import json
import os
import sys

# 以脚本方式运行时 sys.path[0] 是临时目录，这里把项目根（当前工作目录）加回去，
# 否则 from app.config import ... 会 ModuleNotFoundError
sys.path.insert(0, os.getcwd())

out = {"python": sys.version.split()[0], "executable": sys.executable}

try:
    import fastapi, openai, murmurai  # noqa: F401
    out["deps"] = True
except Exception as exc:
    out["deps_error"] = "{}: {}".format(type(exc).__name__, exc)

try:
    import torch
    out["torch"] = torch.__version__
    out["cuda"] = bool(torch.cuda.is_available())
    if out["cuda"]:
        out["cuda_name"] = torch.cuda.get_device_name(0)
        out["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)
except Exception as exc:
    out["torch_error"] = "{}: {}".format(type(exc).__name__, exc)

try:
    import ctranslate2
    out["ct2_devices"] = int(ctranslate2.get_cuda_device_count())
except Exception as exc:
    out["ct2_error"] = "{}: {}".format(type(exc).__name__, exc)

try:
    from app.services.audio import check_ffmpeg
    info = check_ffmpeg()
    out["ffmpeg"] = info["ffmpeg_path"]
    out["ffprobe"] = info["ffprobe_path"]
except Exception as exc:
    out["ffmpeg_error"] = "{}: {}".format(type(exc).__name__, exc)

try:
    from app.config import settings
    settings.ensure_dirs()
    out["data_root"] = str(settings.data_root)
    out["hf"] = bool(settings.hf_token_present)
    out["deepseek"] = bool(settings.deepseek_key_present)
    out["asr_model"] = settings.asr_model
    out["chunk_minutes"] = settings.chunk_minutes
except Exception as exc:
    out["config_error"] = "{}: {}".format(type(exc).__name__, exc)

try:
    from app.services.transcribe import BATCH_SIZE, COMPUTE_TYPE, DEVICE
    out["compute_type"] = COMPUTE_TYPE
    out["batch"] = BATCH_SIZE
    out["device"] = DEVICE
except Exception as exc:
    out["asr_error"] = "{}: {}".format(type(exc).__name__, exc)

print(json.dumps(out, ensure_ascii=False))
'@

try {
    [System.IO.File]::WriteAllText($probePath, $probeCode, (New-Object System.Text.UTF8Encoding $false))
    $raw = & $Python $probePath 2>$null
    $code = $LASTEXITCODE
} finally {
    Remove-Item $probePath -Force -ErrorAction SilentlyContinue
}

if ($code -ne 0 -or -not $raw) {
    Fail "探测脚本执行失败（退出码 $code）"
    Hint "先手动跑一次：$Python -c ""import app.config"" 看具体报错"
    exit 1
}
$info = ($raw -join "") | ConvertFrom-Json

# ---------------------------------------------------------------------------
# 3. Python 版本
# ---------------------------------------------------------------------------
Step "2/8 Python 版本"
$ver = $null
try { $ver = [version]$info.python } catch { $ver = $null }
if ($ver -and $ver -ge [version]"3.10" -and $ver -lt [version]"3.14") {
    Ok "Python $($info.python)"
} else {
    Fail "Python $($info.python) 不在 3.10 ~ 3.13 范围内"
    Hint "murmurai-core 要求 < 3.14；请安装 Python 3.12 并重建 .venv"
}

# ---------------------------------------------------------------------------
# 4. Python 依赖
# ---------------------------------------------------------------------------
Step "3/8 Python 依赖"
if ($info.deps_error) {
    Fail "依赖导入失败：$($info.deps_error)"
    Hint "pip install -r requirements.txt"
} else {
    Ok "fastapi / openai / murmurai 均可导入"
}

# ---------------------------------------------------------------------------
# 5. torch 与 CUDA
# ---------------------------------------------------------------------------
Step "4/8 PyTorch 与 CUDA"
if ($info.torch_error) {
    Fail "torch 不可用：$($info.torch_error)"
    Hint "pip install torch==2.9.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128"
} elseif ($info.cuda) {
    Ok "torch $($info.torch) · CUDA 可用 · $($info.cuda_name) · $($info.vram_gb)GB 显存"
} else {
    Fail "torch $($info.torch) 已安装，但 CUDA 不可用（多半装成了 CPU 版）"
    Hint "pip uninstall -y torch torchaudio 后重装 cu128 版；并确认 nvidia-smi 的 CUDA Version >= 12.8"
}

# ---------------------------------------------------------------------------
# 6. cuDNN（ctranslate2 视角）
# ---------------------------------------------------------------------------
Step "5/8 cuDNN 9（ctranslate2）"
if ($null -ne $info.ct2_devices) {
    if ($info.ct2_devices -ge 1) {
        Ok "ctranslate2 检测到 $($info.ct2_devices) 个 CUDA 设备（cuDNN 9 已就绪）"
    } else {
        Fail "ctranslate2 看不到 CUDA 设备 —— 缺 cuDNN 9 的 DLL，转录会静默退回 CPU"
        Hint "pip install ""nvidia-cudnn-cu12==9.*""，再把 .venv\Lib\site-packages\nvidia\cudnn\bin 加入 PATH"
        Hint "详解见 README B6"
    }
} else {
    Fail "ctranslate2 不可用：$($info.ct2_error)"
    Hint "pip install -r requirements.txt"
}

# ---------------------------------------------------------------------------
# 7. FFmpeg
# ---------------------------------------------------------------------------
Step "6/8 FFmpeg / FFprobe"
if ($info.ffmpeg_error) {
    Fail "FFmpeg 不可用：$($info.ffmpeg_error)"
    Hint "winget install --id Gyan.FFmpeg -e；或在 .env 里写 FFMPEG_BIN=C:/ffmpeg/bin/ffmpeg.exe"
} else {
    Ok "ffmpeg  = $($info.ffmpeg)"
    Ok "ffprobe = $($info.ffprobe)"
}

# ---------------------------------------------------------------------------
# 8. 密钥与 .env
# ---------------------------------------------------------------------------
Step "7/8 密钥（.env）"
$envFile = Join-Path $ProjectRoot ".env"
if (-not (Test-Path $envFile)) {
    Warn "未找到 .env，只会读取系统环境变量"
    Hint "copy .env.example .env  然后填入 HF_TOKEN / DEEPSEEK_API_KEY"
} else {
    $envRaw = Get-Content $envFile -Raw -Encoding UTF8
    if ($envRaw -match 'hf_x{6,}' -or $envRaw -match 'sk-x{6,}') {
        Warn ".env 里仍是模板占位符（hf_xxxx / sk-xxxx），请替换成真实密钥"
    } else {
        Ok ".env 已存在：$envFile"
    }
}
if ($info.hf) {
    Ok "HF_TOKEN 已注入"
} else {
    Fail "HF_TOKEN 未读到"
    Hint "检查 .env 第 1 行；或确认系统环境变量里有 HF_TOKEN"
}
if ($info.deepseek) {
    Ok "DEEPSEEK_API_KEY 已注入"
} else {
    Fail "DEEPSEEK_API_KEY 未读到"
    Hint "检查 .env；密钥在 https://platform.deepseek.com 创建"
}

# ---------------------------------------------------------------------------
# 9. 配置与数据目录
# ---------------------------------------------------------------------------
Step "8/8 配置与数据目录"
if ($info.config_error) {
    Fail "配置加载失败：$($info.config_error)"
} else {
    Ok "data_root = $($info.data_root)"
    Ok "ASR: model=$($info.asr_model) compute=$($info.compute_type) batch=$($info.batch) chunk=$($info.chunk_minutes)min"
    if ($info.vram_gb -and $info.vram_gb -lt 10 -and $info.asr_model -eq "large-v3") {
        Warn "显存仅 $($info.vram_gb)GB，但 ASR_MODEL 仍是 large-v3（约需 10GB+）"
        Hint "在 .env 里改：ASR_MODEL=medium / ASR_COMPUTE_TYPE=int8_float16 / ASR_BATCH_SIZE=4"
    }
}

# 显卡驱动（仅做信息展示）
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    $gpu = & nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>$null
    if ($gpu) { Ok "驱动: $($gpu.Trim())" }
} else {
    Warn "nvidia-smi 不可用（未装 NVIDIA 驱动？）"
}

# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
Write-Host ""
if ($script:fail -eq 0) {
    Write-Host "全部通过（警告 $($script:warn) 项）。" -ForegroundColor Green
    Write-Host "下一步：powershell -ExecutionPolicy Bypass -File .\scripts\start-local.ps1" -ForegroundColor Green
    exit 0
} else {
    Write-Host "失败 $($script:fail) 项、警告 $($script:warn) 项，请按上面的 [FAIL] 提示修复后再启动。" -ForegroundColor Red
    exit 1
}
