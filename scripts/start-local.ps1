#requires -Version 5.1
<#
.SYNOPSIS
    Polyglot Media Summarizer —— 一键启动本地服务（Windows）。

.DESCRIPTION
    自动完成「进目录 → 找 .venv → 检查端口 → 启动 uvicorn」，
    并在服务就绪后自动打开接口文档页面。

    首次使用如果还没建虚拟环境，加 -Setup 让它顺手装好
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
    .\scripts\start-local.ps1

.EXAMPLE
    .\scripts\start-local.ps1 -Port 8001 -Reload

.EXAMPLE
    .\scripts\start-local.ps1 -Setup -BindAddress 0.0.0.0
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
$python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    if (-not $Setup) {
        Write-Host "[FAIL] 未找到虚拟环境：$python" -ForegroundColor Red
        Write-Host "       请先按 README B5 执行：" -ForegroundColor DarkGray
        Write-Host "         py -3.12 -m venv .venv" -ForegroundColor DarkGray
        Write-Host "         .\.venv\Scripts\Activate.ps1" -ForegroundColor DarkGray
        Write-Host "         pip install -r requirements.txt" -ForegroundColor DarkGray
        Write-Host "         pip install torch==2.9.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128" -ForegroundColor DarkGray
        Write-Host "       或让本脚本自动准备（会下载数 GB）：.\scripts\start-local.ps1 -Setup" -ForegroundColor DarkGray
        exit 1
    }

    Write-Host "[..] -Setup：创建虚拟环境并安装依赖（耗时较长，请耐心等）" -ForegroundColor Cyan
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -m venv .venv
    } else {
        & python -m venv .venv
    }
    if (-not (Test-Path $python)) {
        Write-Host "[FAIL] 虚拟环境创建失败，请先装 Python 3.12（README B2）" -ForegroundColor Red
        exit 1
    }
    & $python -m pip install --upgrade pip
    & $python -m pip install -r requirements.txt
    & $python -m pip install torch==2.9.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
    Write-Host "[OK] 依赖安装完成；建议先跑一次 .\scripts\check-env.ps1" -ForegroundColor Green
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
    Write-Host "       换端口：.\scripts\start-local.ps1 -Port 8001" -ForegroundColor DarkGray
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
