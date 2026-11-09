#requires -Version 5.1
<#
.SYNOPSIS
    Polyglot Media Summarizer —— 一键启动本地服务（Windows）。

.DESCRIPTION
    自动完成「进目录 → 找 .venv → 检查端口 → 启动 uvicorn」，
    并在服务就绪后自动打开接口文档页面。

    首次使用如果还没建虚拟环境，加 -Setup 自动准备
    （会下载 torch 等数 GB 依赖，耗时较长）。

.PARAMETER Port
    HTTP 端口，默认 8000。

.PARAMETER BindAddress
    监听地址，默认 127.0.0.1（只允许本机访问）。
    想给局域网其他设备访问用 0.0.0.0（注意：本项目没有鉴权）。

.PARAMETER Reload
    代码改动自动重载，开发时用。

.PARAMETER Setup
    没有 .venv 时自动创建并安装依赖（含 CUDA 版 torch）。

.PARAMETER NoBrowser
    不自动打开浏览器。

.EXAMPLE
    .\scripts\start-server.ps1

.EXAMPLE
    .\scripts\start-server.ps1 -Port 8001 -Reload

.EXAMPLE
    .\scripts\start-server.ps1 -Setup -BindAddress 0.0.0.0
#>
[CmdletBinding()]
param(
    [int]$Port = 8000,
    [string]$BindAddress = "127.0.0.1",
    [switch]$Reload,
    [switch]$Setup,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
$env:PYTHONUTF8 = "1"

Write-Host "==== Polyglot Media Summarizer ====" -ForegroundColor White
Write-Host "项目目录: $ProjectRoot"

# ---------------------------------------------------------------------------
# 1. 定位 / 准备 Python 虚拟环境
# ---------------------------------------------------------------------------
function Find-VenvPython {
    # 兼容 .venv / venv / env 三种目录名
    param([string]$Root)
    foreach ($name in @(".venv", "venv", "env")) {
        $candidate = Join-Path $Root "$name\Scripts\python.exe"
        if (Test-Path $candidate) { return $candidate }
    }
    return $null
}

function Find-CompatiblePythonArg {
    # murmurai-core 支持 3.10 ~ 3.13。py 启动器默认可能指向 3.14（装不上依赖），
    # 所以这里从高到低显式挑一个兼容版本。
    foreach ($v in @("3.13", "3.12", "3.11", "3.10")) {
        $arg = "-$v"
        & py $arg -c "import sys" 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { return $arg }
    }
    return $null
}

$python = Find-VenvPython -Root $ProjectRoot
if (-not $python) { $python = Join-Path $ProjectRoot ".venv\Scripts\python.exe" }

if (-not (Test-Path $python)) {
    if (-not $Setup) {
        Write-Host "[FAIL] 未找到虚拟环境（已查找 .venv / venv）" -ForegroundColor Red
        Write-Host "       请先按 README「本地安装」第 6 节执行：" -ForegroundColor DarkGray
        Write-Host "         rem 看装了哪些 Python（要 3.10~3.13）" -ForegroundColor DarkGray
        Write-Host "         py -0p" -ForegroundColor DarkGray
        Write-Host "         rem 把 3.13 换成你实际有的版本" -ForegroundColor DarkGray
        Write-Host "         py -3.13 -m venv .venv" -ForegroundColor DarkGray
        Write-Host "         rem cmd 激活（PowerShell 里用 .\.venv\Scripts\Activate.ps1）" -ForegroundColor DarkGray
        Write-Host "         .venv\Scripts\activate.bat" -ForegroundColor DarkGray
        Write-Host "         pip install torch==2.9.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128" -ForegroundColor DarkGray
        Write-Host "         pip install -r requirements.txt" -ForegroundColor DarkGray
        Write-Host "       或让本脚本自动准备（会下载数 GB）：scripts\start-server.bat -Setup" -ForegroundColor DarkGray
        exit 1
    }

    Write-Host "[..] -Setup：创建虚拟环境并安装依赖（耗时较长，请耐心等）" -ForegroundColor Cyan
    $venvDir = Join-Path $ProjectRoot ".venv"
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $verArg = Find-CompatiblePythonArg
        if (-not $verArg) {
            Write-Host "[FAIL] 没找到 3.10~3.13 的 Python（3.14 不被 murmurai-core 支持）" -ForegroundColor Red
            Write-Host "       安装 3.12：winget install --id Python.Python.3.12 -e" -ForegroundColor DarkGray
            exit 1
        }
        Write-Host "       使用 Python $verArg 创建虚拟环境" -ForegroundColor DarkGray
        & py $verArg -m venv $venvDir
    } else {
        & python -m venv $venvDir
    }

    $python = Join-Path $venvDir "Scripts\python.exe"
    if (-not (Test-Path $python)) {
        Write-Host "[FAIL] 虚拟环境创建失败，请先装 Python 3.12/3.13（README「本地安装」第 3 节）" -ForegroundColor Red
        exit 1
    }
    $pyVer = & $python -c "import sys; print('%d.%d' % sys.version_info[:2])"
    Write-Host "       venv Python: $pyVer" -ForegroundColor DarkGray
    if ([version]$pyVer -ge [version]"3.14") {
        Write-Host "[FAIL] 虚拟环境是 Python $pyVer，murmurai-core 要求 < 3.14" -ForegroundColor Red
        Write-Host "       删掉重建：rmdir /s /q .venv  然后 py -3.13 -m venv .venv" -ForegroundColor DarkGray
        exit 1
    }

    & $python -m pip install --upgrade pip
    # 顺序很重要：先 CUDA 版 torch，再 requirements；反过来 pip 会先解析出 CPU 版 torch
    & $python -m pip install torch==2.9.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
    & $python -m pip install -r requirements.txt
    & $python -m pip check
    Write-Host "[OK] 依赖安装完成；建议先跑一次 scripts\check-env.bat" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
# 1.5 把 venv 里的 cuDNN 9 补进「本次会话」的 PATH
#   为什么需要：README 第 7 节里那条 `set PATH=...` 只对当时那个窗口有效，关机或新开
#   窗口就没了；而 ctranslate2 找不到 cuDNN 9 时**不会报错，只会静默退回 CPU**
#   （表现为"第一次很快，重启后突然慢十倍"）。
#   - 目录从虚拟环境路径反推，所以 .venv / venv / env 三种命名都能覆盖
#   - 只影响当前进程及其拉起的 uvicorn，不写系统 PATH、不污染其他项目
#   - 没装 nvidia-cudnn-cu12（或已用系统级 cuDNN）时静默跳过
# ---------------------------------------------------------------------------
$venvRoot = Split-Path -Parent (Split-Path -Parent $python)
$cudnnBin = Join-Path $venvRoot "Lib\site-packages\nvidia\cudnn\bin"
if (Test-Path $cudnnBin) {
    if (($env:PATH -split ';') -notcontains $cudnnBin) {
        $env:PATH = "$cudnnBin;$env:PATH"
    }
    Write-Host "[OK] cuDNN 9 已加入本次会话 PATH：$cudnnBin" -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
# 2. .env 检查（没有就提示，不强制阻断）
# ---------------------------------------------------------------------------
$envFile = Join-Path $ProjectRoot ".env"
if (-not (Test-Path $envFile)) {
    Write-Host "[WARN] 未找到 .env，将只读取系统环境变量" -ForegroundColor Yellow
    Write-Host "       建议：copy .env.example .env  然后填入 HF_TOKEN / DEEPSEEK_API_KEY" -ForegroundColor DarkGray
    $answer = Read-Host "仍然启动？(y/N)"
    if ($answer -ne "y" -and $answer -ne "Y") { exit 1 }
}

# ---------------------------------------------------------------------------
# 3. 端口占用检查
# ---------------------------------------------------------------------------
function Test-PortFree {
    param([int]$p)
    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Any, $p)
        $listener.Start()
        return $true
    } catch {
        return $false
    } finally {
        if ($listener) { $listener.Stop() }
    }
}

if (-not (Test-PortFree -p $Port)) {
    Write-Host "[FAIL] 端口 $Port 已被占用" -ForegroundColor Red
    Write-Host "       换端口：scripts\start-server.bat -Port 8001" -ForegroundColor DarkGray
    Write-Host "       查占用：Get-NetTCPConnection -LocalPort $Port | Select-Object OwningProcess" -ForegroundColor DarkGray
    exit 1
}

# ---------------------------------------------------------------------------
# 4. 打开浏览器（等服务真起来再开，最多等 60 秒）
# ---------------------------------------------------------------------------
$openHost = $BindAddress
if ($BindAddress -eq "0.0.0.0") { $openHost = "127.0.0.1" }
$docsUrl = "http://{0}:{1}/docs" -f $openHost, $Port

$job = $null
if (-not $NoBrowser) {
    try {
        $job = Start-Job -ScriptBlock {
            param($url, $p)
            for ($i = 0; $i -lt 120; $i++) {
                $client = $null
                try {
                    $client = New-Object System.Net.Sockets.TcpClient
                    $client.Connect("127.0.0.1", $p)
                    Start-Process $url
                    return
                } catch {
                    Start-Sleep -Milliseconds 500
                } finally {
                    if ($client) { $client.Close() }
                }
            }
        } -ArgumentList $docsUrl, $Port
    } catch {
        Write-Host "[WARN] 无法创建浏览器辅助任务，跳过自动打开（不影响服务）" -ForegroundColor Yellow
    }
}

# ---------------------------------------------------------------------------
# 5. 启动
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "接口文档: $docsUrl" -ForegroundColor Green
Write-Host ("健康检查: http://{0}:{1}/healthz" -f $openHost, $Port) -ForegroundColor Green
Write-Host "产物目录: $(Join-Path $ProjectRoot 'data\outputs')" -ForegroundColor Green
Write-Host "停止服务: Ctrl + C" -ForegroundColor DarkGray
if ($BindAddress -eq "0.0.0.0") {
    Write-Host "注意: 已监听 0.0.0.0，局域网内其他设备可访问，且本项目没有鉴权" -ForegroundColor Yellow
}
Write-Host ""

$uvicornArgs = @("-m", "uvicorn", "app.main:app", "--host", $BindAddress, "--port", "$Port")
if ($Reload) { $uvicornArgs += "--reload" }

try {
    & $python @uvicornArgs
} finally {
    if ($job) {
        Stop-Job $job -ErrorAction SilentlyContinue
        Remove-Job $job -Force -ErrorAction SilentlyContinue
    }
    Write-Host ""
    Write-Host "服务已停止。" -ForegroundColor DarkGray
}
