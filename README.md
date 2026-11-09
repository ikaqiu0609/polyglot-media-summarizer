# Polyglot Media Summarizer

> 上传一段多语言视频/音频，在本机 GPU 上完成转录与说话人分离，得到带时间戳和说话人归属的转录 JSON；确认内容后，再生成中文结构化总结。
> 音频与模型全部在本机运行，只有转录文本会发送给 DeepSeek。

**默认两段式，不会自动产生费用**：转录（本地 GPU，免费）→ `transcribed` 暂停 → 调用 `/summarize` → DeepSeek 总结（付费）。详见 [两段式流程](#两段式流程转录免费总结付费)。

支持 `.mp4 / .mkv / .mov / .webm / .mp3 / .wav / .m4a / .flac ...`，适配中文/英文/韩文/日文等语言的混合视频（如各种国际直播）。

---

## 目录

- [核心特性](#核心特性)
- [架构](#架构)
- [目录结构](#目录结构)
- [模型授权（必读）](#模型授权必读)
- [两段式流程（转录免费，总结付费）](#两段式流程转录免费总结付费)
- [本地安装（Windows）](#本地安装windows)
- [第一次使用](#第一次使用)
- [API 参考](#api-参考)
- [任务状态与进度](#任务状态与进度)
- [产物与文件布局](#产物与文件布局)
- [命令行工具](#命令行工具)
- [配置项](#配置项)
- [性能与显存](#性能与显存)
- [常见问题](#常见问题)
- [已知限制](#已知限制)

---

## 核心特性

| 能力 | 实现 | 说明 |
|---|---|---|
| 音频抽取 | FFmpeg | 任意容器 → 16kHz / 单声道 / PCM16 WAV，`-c:a copy` 切片不重编码 |
| 语音识别 | `murmurai-core`（WhisperX fork）+ `faster-whisper` | 支持 `large-v3` / `large-v3-turbo` / `medium`，按 [显存档位推荐](#显存档位推荐) 选择 |
| 说话人分离 | `pyannote.audio` + `pyannote/speaker-diarization-3.1` | 整段只跑一次，跨切片说话人标签一致 |
| 长音频处理 | FFmpeg 按时长切片 + Silero VAD | 超过 15 分钟自动切分，规避显存溢出 |
| 边界拼接 | 切点吸附静音 + 词级拆段 + 字符级去重 | 切片切点造成的半句/重复词自动修复 |
| 结构化总结 | DeepSeek（OpenAI SDK 兼容） | 分段摘要 → 全局聚合，输出六段式中文 Markdown |
| **费用控制** | **两段式（转录 / 总结解耦）** | **转录免费且本地完成，JSON 落盘后停在 `transcribed`；只有显式调用 `/summarize` 才产生 API 费用** |
| 服务化 | FastAPI + BackgroundTasks | 上传即返回 `task_id`，轮询进度 |
| 运行环境 | Windows 10 / 11 + NVIDIA 显卡 | 最低 4GB（需关闭说话人分离）；推荐 6GB 以上 |
| 配置方式 | 环境变量 + `.env` | 项目根目录的 `.env`（`python-dotenv`） |

---

## 架构

```
                    ┌────────────────────────────────────────────────┐
  POST /upload ───► │  FastAPI  app/main.py                          │
                    │  ├─ 流式落盘 data/uploads/{task_id}.mp4        │
                    │  └─ BackgroundTasks → run_pipeline(task_id)    │
                    └───────────────────┬────────────────────────────┘
                                        │
                    ┌───────────────────▼────────────────────────────┐
                    │ ① audio.py     FFmpeg 16k mono + 按时长切片     │
                    ├────────────────────────────────────────────────┤
                    │ ② transcribe.py       （本地 GPU，不花钱）      │
                    │    pyannote 整段分离 → murmurai 逐片转录        │
                    │    → 全局时间戳 → 边界合并去重                   │
                    └───────────────────┬────────────────────────────┘
                                        │
                     data/outputs/{task_id}.transcript.json
                                        │
                         ┌──────────────▼──────────────┐
                         │ status = transcribed（暂停） │
                         │ 不调用 DeepSeek，不产生费用   │
                         └──────────────┬──────────────┘
                                        │ 显式确认（二选一）
                    ┌───────────────────┴────────────────────┐
                    │ POST /summarize/{task_id}              │
                    │ POST /summarize（上传 transcript.json） │
                    └───────────────────┬────────────────────┘
                                        │
                    ┌───────────────────▼────────────────────────────┐
                    │ ③ summarizer.py      （DeepSeek，按 token 计费）│
                    │    Map：按字数分块 → JSON 局部摘要（校验时间戳）  │
                    │    Reduce：片段索引 → 六段式中文 Markdown        │
                    └───────────────────┬────────────────────────────┘
                                        │
                             data/outputs/{task_id}.md
```

**为什么分两段？** ASR 负责「听清 + 分清谁在说」，产出机器可校验的 JSON；语言理解交给 DeepSeek。更重要的是成本边界：转录用本机 GPU，边际成本为零；DeepSeek 按 token 计费。拆开后可以先完成转录、人工校对 JSON，确认值得再生成总结。`AUTO_SUMMARIZE=true` 可改为转录完成即总结。

---

## 目录结构

```
.
├── requirements.txt             # Python 依赖（torch 需按安装步骤单独装）
├── .env.example                 # 环境变量模板（复制为 .env，已 gitignore）
├── run.bat                      # 日常入口：双击即启动（转调 scripts\start-server.bat）
├── app/
│   ├── config.py                # 环境变量集中读取（Settings）
│   ├── main.py                  # FastAPI 接口 + 后台任务编排
│   └── services/
│       ├── audio.py             # FFmpeg 抽取与切片
│       ├── transcribe.py        # 转录 + 说话人分离 + 边界合并
│       └── summarizer.py        # DeepSeek 两阶段总结
├── scripts/
│   ├── check-env.bat            # 环境自检（cmd / 双击入口，转调 check-env.ps1）
│   ├── start-server.bat         # 一键启动（cmd / 双击入口，转调 start-server.ps1）
│   ├── check-env.ps1            # 自检的实际逻辑
│   └── start-server.ps1         # 启动的实际逻辑（venv + 端口检查 + 开浏览器）
├── tests/
│   └── test_pipeline_utils.py   # 纯函数回归测试
└── data/                        # 运行时生成（已 gitignore）
    ├── uploads/                 # 原始上传文件
    ├── audio/                   # 16k wav / 切片（任务结束自动清理）
    └── outputs/                 # {task_id}.md、{task_id}.transcript.json
```

---

## 模型授权（必读）

> 这一段决定「说话人分离能不能用」。不点 Agree，日志里会反复出现 403，所有段落都会变成 `UNKNOWN`。

### 0. 三个必须 Agree 的 gated 模型

用**同一个** HuggingFace 账号依次打开、登录、点 **Agree and access repository**：

| # | 模型页 | 作用 | 不点会怎样 |
|---|---|---|---|
| 1 | <https://hf.co/pyannote/speaker-diarization-3.1> | 说话人分离主管线（默认 `DIARIZATION_MODEL`） | 403，整份结果无说话人标签 |
| 2 | <https://hf.co/pyannote/segmentation-3.0> | 3.1 使用的分割模型 | 403，同上 |
| 3 | <https://hf.co/pyannote/speaker-diarization-community-1> | `pyannote.audio` 4.x 会额外请求它的 PLDA 文件（`plda/xvec_transform.npz`） | 403，同上（最容易漏） |

然后到 <https://hf.co/settings/tokens> 创建 **Read** 权限的 token（`hf_...`），填进 `.env` 的 `HF_TOKEN`。

三个常见坑：

- 三个页面必须用**同一个账号**点 Agree，token 也必须是这个账号的；
- 点完 Agree 后**重启服务**：进程内会缓存「分离不可用」状态，避免每个任务都重新下载权重再失败；
- 企业/组织账号可能需要管理员审批，`403 ... not in the authorized list` 通常就是这个原因。

> 日志里出现 `Cannot access gated repo for url ...` 时，直接看 URL 里的仓库名，回上表确认是否漏点。

### 1. 不需要 Agree、但会自动下载的模型

| 模型 | 何时下载 | 大小 |
|---|---|---|
| `Systran/faster-whisper-<ASR_MODEL>` | 首次转录 | `medium` ≈ 1.5GB，`large-v3-turbo` ≈ 1.6GB，`large-v3` ≈ 3GB |
| `snakers4/silero-vad` | 首次转录（`VAD_METHOD=silero`） | 约 2MB |
| `jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn` | 中文内容且 `ASR_ALIGN` 生效 | 约 1.2GB |
| `kresnik/wav2vec2-large-xlsr-korean` | 韩语内容且 `ASR_ALIGN` 生效 | 约 1.2GB |
| `pyannote/*` 权重 | 首次说话人分离 | 数十 MB |

> 词级对齐只用于「按说话人拆段」，因此**没有说话人标签时一律跳过**（不论 `ASR_ALIGN` 取值）。分离失败时跳过它能省下每种语言约 1.2GB 的下载。

---

## 两段式流程（转录免费，总结付费）

默认 `AUTO_SUMMARIZE=false`。完整流程：

```text
POST /upload
   → extracting → transcribing → transcribed（停住）
   → data/outputs/{task_id}.transcript.json 已落盘
   → 到这里为止 0 API 费用；只想要转录文本可以直接结束
   → 需要总结时，二选一：
       POST /summarize/{task_id}          复用该任务的转录
       POST /summarize  -F file=@x.json   导入已有的转录 JSON
   → summarizing → completed
   → data/outputs/{task_id}.md
```

| 场景 | 推荐操作 |
|---|---|
| 只想要转录文本 / 字幕 | 停在 `transcribed`，直接读 `.transcript.json` |
| 确认内容值得总结 | `POST /summarize/{task_id}` |
| 复用历史转录 | `POST /summarize` 上传 JSON |
| 想让服务全自动 | `.env` 里设 `AUTO_SUMMARIZE=true` |
| 总结失败（402 / 网络） | 修复后再次 `POST /summarize/{task_id}`，不会重新转录 |
| 转录失败，但上传文件还在 | `POST /transcribe/{task_id}` 重新转录 |

命令行示例（Windows cmd）：

```bat
rem 1) 上传，记下 task_id
curl.exe -s -X POST http://127.0.0.1:8000/upload -F "file=@.\sample.mp4"

rem 2) 等 status 变成 transcribed
curl.exe -s http://127.0.0.1:8000/status/<task_id>

rem 3) 触发总结
curl.exe -s -X POST http://127.0.0.1:8000/summarize/<task_id>

rem 或者：导入一个现成的 transcript.json
curl.exe -s -X POST http://127.0.0.1:8000/summarize -F "file=@.\demo.transcript.json"
```

> `POST /summarize/{task_id}` 只允许在 `transcribed` 或 `failed`（且 JSON 还在）时调用：`queued/extracting/diarizing/transcribing` 返回 409，`completed` 且 `.md` 已存在也返回 409。

---

## 本地安装（Windows）

> 目标：从零在一台 Windows 电脑上跑起来。按顺序做，每步都有验证命令，过了再下一步。

### 1. 准备清单

| 项 | 要求 | 怎么确认 |
|---|---|---|
| 系统 | Windows 10 / 11 64 位 | `winver` |
| 显卡 | NVIDIA，显存 ≥ 4GB；推荐 6GB 以上（分离约需 2GB） | `nvidia-smi` |
| 驱动 | 支持 CUDA 12.8（`nvidia-smi` 右上角 `CUDA Version` ≥ 12.8） | 低于 12.8 先去官网升级驱动 |
| 磁盘 | ≥ 15GB 空闲（torch 约 3GB + 模型 3~6GB + 数据） | 资源管理器 |
| 软件 | Git、Python（3.10~3.13，**不要 3.14**）、FFmpeg | 见第 2~4 步 |

> 全程用 **CMD（命令提示符）**：按 `Win` 键 → 输入 `cmd` → 回车。
> 装完任何软件都要**关掉再重开**窗口，否则 PATH 不生效。
>
> ⚠️ **别在 cmd 里敲 `.ps1`**：cmd 不认识 `.ps1`，会弹出「选择一个应用以打开此 .ps1 文件」。记住对应关系：**cmd 用 `activate.bat` / `scripts\*.bat`，PowerShell 才用 `Activate.ps1` / `scripts\*.ps1`**。下面所有命令都是 cmd 版。

### 2. 安装 Git

```bat
winget install --id Git.Git -e --source winget
git --version
```

- 成功标志：`git version 2.x.x`
- 没有 `winget`：去 <https://git-scm.com/download/win> 下载安装包，一路 Next 即可。

### 3. 准备 Python（3.10 ~ 3.13）

先看机器上已经有什么，**有 3.10~3.13 就不用再装**：

```bat
py -0p
```

```
 -V:3.14 *        C:\Users\you\AppData\Local\Programs\Python\Python314\python.exe
 -V:3.13          C:\Users\you\AppData\Local\Programs\Python\Python313\python.exe
```

- `*` 表示 `py` 的**默认**版本。上面这种情况默认是 3.14，**不能用来建虚拟环境**，后面必须显式写 `py -3.13`。
- 只有 3.14 或没有可用版本时：

```bat
winget install --id Python.Python.3.12 -e --source winget

rem 重新确认，装完可能需要新开终端
py -0p
```

- 成功标志：`py -0p` 里出现 `-V:3.12` 或 `-V:3.13`。
- 用官网安装包时，**第一页必须勾选 `Add python.exe to PATH`**。
- 不要装 Microsoft Store 版（路径特殊，虚拟环境容易出问题）。
- **3.14 不行**：`murmurai-core` 要求 `Python < 3.14`，用 3.14 建的环境装依赖会直接报 `requires Python <3.14`。

### 4. 安装 FFmpeg

```bat
winget install --id Gyan.FFmpeg -e --source winget
ffmpeg -version
```

- 成功标志：第一行 `ffmpeg version 7.x`
- 装了但仍提示「不是内部或外部命令」：重开窗口（PATH 才会刷新）；还不行就手动下载 <https://www.gyan.dev/ffmpeg/builds/> 的 `release-essentials` 压缩包，解压到 `C:\ffmpeg`，然后二选一：
  - 把 `C:\ffmpeg\bin` 加进系统 PATH；
  - 或者稍后在 `.env` 里写 `FFMPEG_BIN=C:/ffmpeg/bin/ffmpeg.exe`。

### 5. 克隆仓库

```bat
cd /d "%USERPROFILE%\Documents"
git clone <仓库地址> polyglot-media-summarizer
cd polyglot-media-summarizer
dir
```

- 成功标志：`dir` 能看到 `requirements.txt`、`app`。
- 私有仓库会提示输入账号密码或访问令牌。
- 小技巧：在文件资源管理器进入该文件夹，地址栏输入 `cmd` 回车，终端就直接开在这个目录。

### 6. 创建虚拟环境并安装依赖

```bat
rem 下面这条的 3.13 换成第 3 步 py -0p 看到的版本（3.10 ~ 3.13）
py -3.13 -m venv .venv

rem cmd 激活脚本是 activate.bat
.venv\Scripts\activate.bat

rem 必须显示所选版本，且提示符前出现 (.venv)
python -V

python -m pip install --upgrade pip

rem ① 先装 CUDA 版 torch（顺序不能反）
pip install torch==2.9.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

rem ② 再装项目依赖
pip install -r requirements.txt
pip check
```

激活时报「禁止运行脚本」——这只会在 **PowerShell** 里出现（cmd 用的是 `activate.bat`，不受执行策略影响）。真要用 PowerShell 就执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

或者不激活，直接指定虚拟环境里的解释器：

```bat
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

**为什么 torch 必须放在 requirements 前面？**

- `pip install -r requirements.txt` 会因为 `murmurai-core` 的 `torch>=2.9.0` 间接装一个 PyPI 默认版 torch（Windows 上是 CPU 版）；
- 先装 cu128 版，第 ② 步会认为「已满足」，不会把 CUDA 版覆盖回 CPU 版；
- 顺序反了会留下 torchvision / torchcodec 与新 torch 不配套，`pip` 报 `torchvision ... requires torch>=x.y`。

- 成功标志：`2.9.0+cu128 12.8 True`
- 若是 `2.9.0+cpu None False`：装成 CPU 版了，重跑 ① 那条。
- 若提示 `torchvision ... requires torch>=...`：本项目不使用 torchvision，执行 `pip uninstall -y torchvision` 即可。
- 若提示 `UnicodeDecodeError: 'gbk' codec ...`：`requirements.txt` 被保存成了「中文注释 + 无 BOM」。仓库版本是纯 ASCII，不会出现；自己加过中文注释的话，先 `set PYTHONUTF8=1` 再装。

### 7. 让 ctranslate2 找到 cuDNN 9

`torch.cuda.is_available()` 为 True 只说明 PyTorch 能用 GPU；转录走的是 ctranslate2，它需要单独找到 cuDNN 9 的 DLL，否则**静默退回 CPU**（不报错，只是慢十倍）。

先判断要不要做这一步：

```bat
python -c "import ctranslate2; print('cuda devices:', ctranslate2.get_cuda_device_count())"
```

- 输出 `>= 1`：已有可用 cuDNN，**直接跳到第 8 步**。
- 输出 `0`：按下表三选一。

| 方案 | 做法 | 说明 |
|---|---|---|
| **A. pip 安装（推荐）** | 见下方命令 | 随虚拟环境一起删除，干净 |
| B. 复用 PyTorch 自带 DLL | 先 `dir .\.venv\Lib\site-packages\torch\lib\cudnn_*.dll` 确认有文件，再把该目录加进 PATH | 免下载；目录里 DLL 多，可能与其他 CUDA 组件互相干扰 |
| C. 系统级安装 | 从 NVIDIA 官网下 cuDNN 9，解压后把 `bin` 加进 PATH | 多项目共用，需自己管版本 |

方案 A 的完整命令：

```bat
pip install "nvidia-cudnn-cu12==9.*"

rem 仅当前 cmd 窗口有效（在项目根目录执行，%CD% 就是项目根）
set "PATH=%CD%\.venv\Lib\site-packages\nvidia\cudnn\bin;%PATH%"

rem 想永久生效：打开「用户变量」→ Path → 编辑 → 新建 → 粘贴绝对路径：
rem     D:\你的路径\polyglot-media-summarizer\.venv\Lib\site-packages\nvidia\cudnn\bin
rundll32 sysdm.cpl,EditEnvironmentVariables

rem 不要用 setx 改 PATH：超过 1024 字符会被截断
```

> PowerShell 等价写法：`$env:PATH = "$PWD\.venv\Lib\site-packages\nvidia\cudnn\bin;$env:PATH"`。
> `set` / `$env:` 都只影响**当前窗口**，关掉就失效。
>
> ✅ **用 `run.bat` 启动就不用管这些**：启动脚本会自动把 `.venv\Lib\site-packages\nvidia\cudnn\bin` 加进本次会话的 PATH。

验证（两条都要过）：

```bat
python -c "import ctypes; ctypes.WinDLL('cudnn_ops64_9.dll'); print('cudnn 9 DLL 可加载')"
python -c "import ctranslate2; print('cuda devices:', ctranslate2.get_cuda_device_count())"
```

### 8. 填写 `.env`

```bat
copy .env.example .env
notepad .env
```

| 变量 | 填什么 | 说明 |
|---|---|---|
| `HF_TOKEN` | `hf_...` | 必填；需为 [3 个 gated 模型](#0-三个必须-agree-的-gated-模型) 点过 Agree 的账号 |
| `DEEPSEEK_API_KEY` | `sk-...` | 只有生成总结时才会用到 |
| `AUTO_SUMMARIZE` | `false` | 保持 false：转录完先停下，确认后再决定是否总结 |
| `ASR_MODEL` | `large-v3-turbo` | 按 [显存档位推荐](#显存档位推荐) 选择 |
| `ASR_COMPUTE_TYPE` | `float16` | 显存不足时改 `int8_float16` |
| `ASR_BATCH_SIZE` | `4` | 起始值；OOM 时服务会自动减半重试 |
| `ASR_ALIGN` | `1` | 分离失败时自动跳过词级对齐；显存紧张可设 `0` |
| `CHUNK_MINUTES` | `10` | 影响时间戳精度与重跑粒度，对峰值显存影响很小 |
| `DATA_ROOT` | `data` | 相对路径按项目根解析，不要写 `C:\...` |

- 其他显存档位的推荐组合见 [显存档位推荐](#显存档位推荐)。
- 记事本保存即可（`app/config.py` 按 UTF-8 读取）。
- 路径类配置支持 `D:/data`、`%USERPROFILE%/data`、`data`（相对项目根）。

### 9. 启动服务

```bat
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- 成功标志：`Uvicorn running on http://127.0.0.1:8000`
- 首次转录会按所选模型下载权重（`medium` / `large-v3-turbo` 约 1.5~1.6GB，`large-v3` 约 3GB），连同 pyannote 权重默认存 `%USERPROFILE%\.cache\huggingface`；想换盘就在 `.env` 里设绝对路径 `HF_HOME=D:/models/huggingface`。
- **不要加 `--workers`**：任务状态在进程内存里。

### 10. 验证

浏览器打开 <http://127.0.0.1:8000/docs>：

1. 展开 `POST /upload` → **Try it out** → 选一个 mp4/mkv → **Execute**；
2. 复制返回的 `task_id`；
3. 展开 `GET /status/{task_id}` → 粘贴 id → Execute，反复点到 `status` 变成 `completed`；
4. 浏览器打开 `http://127.0.0.1:8000/result/<task_id>` 下载 Markdown。

命令行等价写法：

```bat
curl.exe -s http://127.0.0.1:8000/healthz
curl.exe -s -X POST http://127.0.0.1:8000/upload -F "file=@.\sample.mp4"
curl.exe -s http://127.0.0.1:8000/status/<task_id>
```

> **PowerShell** 里 `curl` 是 `Invoke-WebRequest` 的别名、不支持 `-F`，所以统一写 `curl.exe`。

### 11. 关闭与下次启动

关闭：在运行服务的窗口按 `Ctrl + C`（或直接关窗口）。

**下次启动（推荐）**

关机重开后不需要重装任何东西：不用再 `pip install`、不用重建 `.venv`、模型也不会重新下载。只要：

1. 打开项目文件夹；
2. 双击 **`run.bat`**。

等价于在 cmd 里：

```bat
cd /d "%USERPROFILE%\Documents\polyglot-media-summarizer"
run.bat
```

脚本会自动完成：定位 `.venv`（不用激活）→ 把 venv 里的 cuDNN 9 加进本次会话 PATH → 检查端口 → 启动 uvicorn → 打开 `/docs`。

**手动启动**

```bat
cd /d "%USERPROFILE%\Documents\polyglot-media-summarizer"
.venv\Scripts\activate.bat
set "PATH=%CD%\.venv\Lib\site-packages\nvidia\cudnn\bin;%PATH%"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

> 第 3 行只在没有做永久 PATH 时需要（见第 7 步）。

### 12. 一键脚本（可选）

仓库自带两组脚本，把安装验证与启动步骤打包好了。**cmd 用 `.bat`，PowerShell 用 `.ps1`**：

| 脚本 | 作用 | cmd 用法 | PowerShell 用法 |
|---|---|---|---|
| `run`（项目根目录） | 日常入口，等价于 `scripts\start-server.bat` | `run.bat` | `.\run.bat` |
| `check-env` | 8 项自检：Python 版本 / 依赖 / torch-CUDA / cuDNN / FFmpeg / 密钥 / 数据目录 / 显存与模型档位 | `scripts\check-env.bat` | `.\scripts\check-env.ps1` |
| `start-server` | 定位虚拟环境 → 检查端口 → 启动 uvicorn → 打开 `/docs` | `scripts\start-server.bat` | `.\scripts\start-server.ps1` |

> `.bat` 只是外壳（绕过执行策略后调用同名 `.ps1`），逻辑在 `.ps1` 里。
> 两个 `.bat` 都能在资源管理器里双击运行；`check-env.bat` 结束时窗口会停住，方便看结果。

常用参数：

```bat
rem 换端口
scripts\start-server.bat -Port 8001

rem 开发模式，改代码自动重载
scripts\start-server.bat -Reload

rem 不自动开浏览器
scripts\start-server.bat -NoBrowser

rem 首次使用：自动挑 3.10~3.13 建 .venv、装 torch + 依赖（会下载数 GB）
scripts\start-server.bat -Setup

rem 局域网访问（本项目无鉴权，谨慎）
scripts\start-server.bat -BindAddress 0.0.0.0
```

自检输出示例（显卡与版本以实际机器为准）：

```
==== Polyglot Media Summarizer · 环境自检 ====
项目目录: D:\code\polyglot-media-summarizer

1/8 Python 解释器
  [OK]   使用虚拟环境: D:\code\polyglot-media-summarizer\.venv\Scripts\python.exe
4/8 PyTorch 与 CUDA
  [OK]   torch 2.9.0+cu128 · CUDA 可用 · NVIDIA GeForce RTX xxxx · xxGB 显存
5/8 cuDNN 9（ctranslate2）
  [OK]   ctranslate2 检测到 1 个 CUDA 设备（cuDNN 9 已就绪）
8/8 配置与数据目录
  [OK]   ASR: model=large-v3-turbo compute=float16 batch=4 chunk=10.0min

全部通过（警告 0 项）。
```

> - 需要 PowerShell 5.1+（Windows 10/11 自带）；`.ps1` 保存为 **UTF-8 with BOM**，避免 5.1 下中文乱码。
> - `.bat` 已内置 `-ExecutionPolicy Bypass`，cmd 用户不需要改策略。
> - `check-env` 有任何 `[FAIL]` 都会以退出码 1 结束。

### 13. 安装排错

| 现象 | 原因 | 解决 |
|---|---|---|
| 敲 `Activate.ps1` 弹出「选择一个应用以打开此 .ps1 文件」 | 在 cmd 里执行了 PowerShell 脚本 | 改用 `.venv\Scripts\activate.bat`，或换到 PowerShell |
| `无法加载文件 Activate.ps1，因为在此系统上禁止运行脚本` | PowerShell 执行策略限制 | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`，或直接 `activate.bat` |
| `python` / `py` 不是内部或外部命令 | 没勾 Add to PATH，或没重开终端 | 重装 Python 并勾选 PATH，或重开窗口 |
| `No suitable Python runtime found` | 机器上没装对应版本 | `py -0p` 看实际版本，再用已装版本建 venv |
| `pip` 报 `UnicodeDecodeError: 'gbk' codec ...` | pip 24.x 在中文 Windows 上按 cp936 解析 requirements | `set PYTHONUTF8=1` 后重试 |
| `pip` 报 `torchvision ... requires torch>=x.y` | 旧环境遗留的 torchvision 不配套 | `pip uninstall -y torchvision`，再 `pip check` |
| `torch.cuda.is_available()` 是 False | 装成了 CPU 版 | 按第 6 步重装 cu128 版 |
| `ct2 cuda devices: 0` | 缺 cuDNN 9 DLL | 按第 7 步处理 |
| 转录极慢、GPU 占用接近 0 | 缺少 cuDNN 9 DLL，已退回 CPU | 用 `run.bat` 启动（自动注入 PATH），或按第 7 步安装 |
| `未找到可执行文件 'ffmpeg'` | 没进 PATH | 按第 4 步设置 `FFMPEG_BIN` |
| `OSError: [WinError 1455] 页面文件太小` | 虚拟内存不足 | 系统属性 → 高级 → 性能设置 → 高级 → 虚拟内存，调到 16GB 以上 |
| 启动报 `[Errno 10048]` 端口占用 | 8000 被别的程序占了 | 换 `--port 8001` |
| 日志中文乱码 | 控制台代码页不是 UTF-8 | 先执行 `chcp 65001` |
| `CUDA out of memory` | 参数超出显存 | 依次降：`ASR_BATCH_SIZE=2` → `ASR_MODEL=medium` → `ENABLE_DIARIZATION=false` |
| 状态一直停在 `transcribed`，没有 `.md` | 默认两段式，等待确认 | 要总结就 `POST /summarize/{task_id}`；想全自动设 `AUTO_SUMMARIZE=true` |
| 日志出现 `No active speech found in audio` | 该切片是静音/纯音乐 | 正常；每片都是这行时检查文件是否真的没有音轨或音量过低 |

---

## 第一次使用

流程：上传 → 抽取音频 → 说话人分离 → 转录 → **停在 `transcribed`** → 确认 → 总结 → 下载。

### 用浏览器（推荐新手）

1. 打开 `http://127.0.0.1:8000/docs`。
2. `POST /upload` → Try it out → 选择文件 → Execute → 记下 `task_id`。
3. `GET /status/{task_id}` → 反复 Execute，直到 `status` 变成 **`transcribed`**（`progress` 约 75）。
4. 此时 `data/outputs/{task_id}.transcript.json` 已经生成。
5. 需要总结时执行 `POST /summarize/{task_id}` → Execute → 等 `status` 变成 `completed`。
6. `GET /result/{task_id}` → Execute → 点「Download file」拿到 Markdown。

### 用命令行

```bash
# 1) 上传
curl -s -X POST http://127.0.0.1:8000/upload -F "file=@./sample.mp4"
# {"task_id":"3b5e...","status":"queued","status_url":"/status/3b5e...","result_url":"/result/3b5e..."}

# 2) 轮询到 transcribed
curl -s http://127.0.0.1:8000/status/3b5e...
# {"status":"transcribed","stage":"转录完成（待总结）","progress":75.0,"detail":"转录完成，等待生成总结", ...}

# 3) 触发总结
curl -s -X POST http://127.0.0.1:8000/summarize/3b5e...

# 4) 下载
curl -OJ http://127.0.0.1:8000/result/3b5e...
```

### 结果在哪

```
data/outputs/{task_id}.transcript.json    ← 转录原文（停在这一步也完全可用）
data/outputs/{task_id}.md                 ← 总结（触发 /summarize 后才生成）
data/uploads/{task_id}.mp4                ← 原始上传文件
```

### 要等多久

见 [性能与显存](#性能与显存)。转录耗时主要取决于有效语音时长、显卡与所选模型：先按档位表选配置，再参考日志里的实际耗时与峰值微调。总结另算（几秒到几十秒），默认需要手动触发。

---

## API 参考

### `POST /upload`

`multipart/form-data`，响应码 `202`。

| 字段 | 类型 | 说明 |
|---|---|---|
| `file` | file | 必填，媒体文件 |
| `speakers` | string（数字） | 可选（填 1~20 的整数），已知说话人数，作为 pyannote 的 min/max；**留空即自动判断**（Swagger UI 里默认就是空框，不会替你填 1） |

响应字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `task_id` | string | 任务 ID（32 位 hex） |
| `status` | string | 固定 `queued` |
| `filename` | string | 原始文件名（已剥离路径） |
| `size_bytes` | int | 实际落盘字节数 |
| `status_url` / `result_url` | string | 后续查询地址 |

错误：`415` 不支持的扩展名、`413` 超过 `MAX_UPLOAD_MB`、`400` 空文件、`422` `speakers` 不是 1~20 的整数。

### `POST /transcribe/{task_id}`

用已上传的媒体重新转录。

| 场景 | 响应 |
|---|---|
| `failed` / `transcribed` / `completed`，且上传文件存在 | `202`，重新排队（`status=queued`） |
| `queued` / `extracting` / `diarizing` / `transcribing` / `summarizing` | `409` 正在处理中 |
| 任务来自导入的转录 JSON（无媒体文件） | `409` |
| 上传文件已被清理 | `410` |

### `POST /summarize/{task_id}`

对已转录任务生成总结（调用 DeepSeek，产生费用）。响应码 `202`，字段含 `task_id`、`status`、`transcript_path`、`status_url`、`result_url`、`note`。

| 场景 | 响应 |
|---|---|
| `transcribed` 或 `failed`（且 JSON 存在） | `202`，开始生成 |
| `queued` / `extracting` / `diarizing` / `transcribing` | `409` 任务尚未转录完成 |
| `summarizing` | `202`，正在总结中 |
| `completed` 且 `.md` 已存在 | `409` 总结已存在 |

### `POST /summarize`

导入现成的 `transcript.json` 并生成总结。`multipart/form-data`，字段名 `file`，上限 50MB。

接受完整转录对象（含 `segments` / `speakers` / `duration_sec` / `diarized`），或直接的 segment 数组。缺 `segments`、文本全空、时间戳无法解析时返回 `400`。

响应 `202`，返回新的 `task_id`，后续用 `/status`、`/result` 跟踪。

### `GET /status/{task_id}`

| 字段 | 说明 |
|---|---|
| `status` | `queued / extracting / diarizing / transcribing / transcribed / summarizing / completed / failed` |
| `stage` | 中文阶段名，如「说话人分离」 |
| `progress` | 0~100，一位小数 |
| `detail` | 当前细节，如「转录切片 3/7」 |
| `elapsed_sec` | 从开始处理到现在的秒数 |
| `error` | 失败原因（仅 `failed` 有值） |
| `result` | 完成后包含 `summary_path`、`transcript_path`、`num_segments`、`speakers`、`tokens` 等 |

错误：`404` 任务不存在。

### `GET /result/{task_id}`

返回 `text/markdown; charset=utf-8`，`Content-Disposition: attachment; filename="summary-{task_id}.md"`。
错误：`404` 不存在、`409` 未完成（`transcribed` 会提示先调用 `/summarize/{task_id}`）、`410` 产物已被清理。

### `GET /healthz`

`ffmpeg` / `ffprobe` 是否可用、脱敏后的配置快照、内存中任务数、等待总结的任务数（`pending_summary`）、媒体解码依赖（`media_io.soundfile` / `media_io.torchcodec`）、说话人分离状态（`diarization.input_mode` / `available` / `cached_reason` / `last_error`）。不返回任何密钥。

---

## 任务状态与进度

| 状态 | 进度区间 | 触发点 |
|---|---|---|
| `queued` | 0% | 已入队，等待 GPU（`MAX_CONCURRENT_TASKS` 串行） |
| `extracting` | 2 → 15% | `audio.analyze()` 抽取 + 切片 |
| `diarizing` | 15 → 35% | 加载 pyannote + 整段说话人分离（失败会降级，不阻断） |
| `transcribing` | 35 → 75% | 逐片转录 + 说话人指派 |
| **`transcribed`** | **75%** | **转录完成、JSON 已落盘，等待生成总结（默认停留点）** |
| `summarizing` | 75 → 98% | 调用 DeepSeek Map → Reduce |
| `completed` | 100% | Markdown 已落盘 |
| `failed` | 停在失败点 | `error` 带原因；`result.summarize_url` 可重试总结，`result.transcribe_url` 可重新转录 |

状态不可回退：进入 `completed` / `failed` 后，迟到的进度回调会被忽略；`transcribed` 也会忽略转录阶段的迟到回调。

> `transcribed` 不是失败，也不是完成。`/result` 此时返回 409，但 `.transcript.json` 已经可以使用。

---

## 产物与文件布局

```
data/
├── uploads/   {task_id}.mp4                       # 原始上传，默认保留（便于重新转录/排查）
├── audio/     {task_id}/xxx.16k.wav  + .partNNN.wav   # 任务结束自动删除
└── outputs/   {task_id}.transcript.json           # 转录原文（transcribed 时就有）
               {task_id}.md                       # 中文总结（/summarize 成功后生成）
```

转录 JSON 结构（也可以直接喂给 `POST /summarize` 或命令行工具）：

```json
{
  "language": "en",
  "chunk_languages": ["en", "en", "ja"],
  "segments": [{"start": 12.34, "end": 18.9, "speaker": "Speaker 1", "text": "..."}],
  "speakers": ["Speaker 1", "Speaker 2"],
  "speaker_stats": [{"speaker": "Speaker 1", "seconds": 320.5, "turns": 42}],
  "num_segments": 128,
  "duration_sec": 3721.5,
  "diarized": true,
  "diarization": {"requested": true, "ok": true, "input_mode": "waveform", "reason": null},
  "speech_ratio": 0.72,
  "peak_vram_mb": 5217.0,
  "chunks": [{"index": 0, "start": 0.0, "duration": 600.0, "language": "en", "segments": 22}],
  "elapsed_sec": 412.7
}
```

---

## 命令行工具

三个模块都支持独立运行，便于分段定位问题（不需要起服务）。

```bash
# ① 只测音频：抽取 + 切片（强制 0.2 分钟 = 12s 一片）
python -m app.services.audio ./sample.mp4 --chunk-minutes 0.2

# ② 只测 ASR（跳过说话人分离，无需 HF_TOKEN）
python -m app.services.transcribe ./sample.mp4 --no-diarize --print 5

# ③ 完整转录（含说话人），导出 JSON
python -m app.services.transcribe ./sample.mp4 --out ./demo.json --speakers 2

# ④ 只测总结（复用上一步 JSON，不占 GPU）
python -m app.services.summarizer ./demo.json --out ./summary.md
```

> ④ 与 `POST /summarize` 等价：都只读转录 JSON，不会再跑一遍 ASR。
> **Windows** 下把 `./sample.mp4` 换成 `.\sample.mp4`；需要临时设变量时，cmd 用 `set HF_TOKEN=hf_xxx`，PowerShell 用 `$env:HF_TOKEN="hf_xxx"`。

输出示例：

```
[   12.34 -    18.90] Speaker 1  So the first thing we need to check is the build pipeline
[   19.10 -    24.55] Speaker 2  Right, and OverlayFS was the bottleneck last time
```

---

## 配置项

全部通过环境变量注入，`app/config.py` 是唯一读取入口。本地开发可以把它们写进**项目根目录的 `.env`**（模板见 `.env.example`）；加载规则是系统环境变量优先、`.env` 兜底，文件不存在也不报错。

路径类配置（`DATA_ROOT` / `UPLOAD_DIR` / `AUDIO_DIR` / `OUTPUT_DIR`）支持三种写法：相对路径（**按项目根解析，与启动目录无关**）、绝对路径（`D:/data`）、带变量（`%USERPROFILE%/data`）。

### 必填

| 变量 | 说明 |
|---|---|
| `HF_TOKEN` | Hugging Face Token，用于加载 pyannote（也接受 `HUGGINGFACE_TOKEN` / `HF_HUB_TOKEN`）；需为 [3 个 gated 模型](#0-三个必须-agree-的-gated-模型) 点过 Agree |
| `DEEPSEEK_API_KEY` | DeepSeek API Key；只在需要总结时必填，缺失时 `/summarize` 会失败（转录不受影响） |

### 服务与文件

| 变量 | 默认 | 说明 |
|---|---|---|
| `DATA_ROOT` | `<repo>/data` | 数据根目录；相对路径按项目根解析 |
| `UPLOAD_DIR` / `AUDIO_DIR` / `OUTPUT_DIR` | 各子目录 | 可分别覆盖 |
| `FFMPEG_BIN` / `FFPROBE_BIN` | `ffmpeg` / `ffprobe` | 可填绝对路径（如 `C:/ffmpeg/bin/ffmpeg.exe`）；留空则走 PATH 自动探测 |
| `MAX_UPLOAD_MB` | `2048` | 上传上限，流式校验 |
| `MAX_CONCURRENT_TASKS` | `1` | 并发任务数，显存受限建议保持 1 |
| `TASK_RETENTION_HOURS` | `24` | 终态任务保留期，超期自动清理内存与产物 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `ALLOWED_EXTENSIONS` | 常见音视频后缀 | 逗号分隔追加 |

### ASR / 说话人分离

| 变量 | 默认 | 说明 |
|---|---|---|
| `ASR_MODEL` | `large-v3` | 按 [显存档位推荐](#显存档位推荐) 选择；显存不足时优先 `large-v3-turbo` 或 `medium` |
| `ASR_LANGUAGE` | `auto` | 英文夹韩日建议 `auto`；强制单语设 `en` |
| `ASR_COMPUTE_TYPE` | `float16` | 显存不足时用 `int8_float16`；CPU 下自动降级 `int8` |
| `ASR_BATCH_SIZE` | `16` | OOM 时自动减半重试，最多 2 次 |
| `VAD_METHOD` | `silero` | 免额外许可，但首次要从 GitHub（torch.hub）下载，网络受限时改 `pyannote`（走 HF，需接受 `segmentation-3.0` 条款）；silero 失败会自动回退 pyannote 重试一次 |
| `ASR_ALIGN` | `1` | 词级对齐开关（`0` 关闭）。对齐只用于「按说话人拆段」，没有说话人标签时一律跳过 |
| `ASR_OFFLINE` | `auto` | `auto`=联网失败时自动切 HF 离线模式重试一次；`1`=始终离线（模型已缓存时更快）；`0`=始终联网 |
| `ASR_LANG_VOTE_SAMPLES` | `3` | 前 N 个「有语音」切片参与语言投票 |
| `ASR_LANG_MIN_SHARE` | `0.8` | 某语言占比达到该阈值才锁定全局语言；多语内容/平票继续逐片检测，顶层 `language` 写 `multilingual` |
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization-3.1` | 需 Agree 的模型见 [模型授权](#模型授权必读) |
| `DIAR_AUDIO_INPUT` | `waveform` | `waveform`=自己解码 16k wav 传入（不依赖 torchcodec，推荐）；`path`=把路径交给 pyannote（需 torchcodec 与 torch 版本匹配 + 共享版 FFmpeg） |
| `ENABLE_DIARIZATION` | `true` | 关掉则只转录、不打说话人标签（可省 HF_TOKEN） |
| `DIAR_MIN_SPEAKERS` / `DIAR_MAX_SPEAKERS` | 空 | 已知人数时设置，显著提升准确率；`/upload` 的 `speakers` 表单字段等价 |
| `DIAR_TRIM_SILENCE` | `1` | 分离前裁掉首尾静音；整段无语音时直接跳过 |
| `DIAR_TRIM_MIN_SEC` / `DIAR_ENERGY_THRESHOLD` | `1.0` / `0.001` | 裁剪后至少保留的语音时长、能量阈值（≈ -60dBFS） |
| `DIAR_MIN_SPEAKER_SEC` / `DIAR_MIN_SPEAKER_SHARE` | `5` / `0.01` | 低于该时长的短时说话人并入最近的保留说话人；占比按有效语音时长计算 |
| `MAX_SPEAKER_ATTACH_GAP` | `2.0` | segment 与最近说话人时间片超过该间隔就标 `UNKNOWN` |
| `DIAR_CACHE_FAILURE` | `1` | 只缓存永久性失败（403/token）；临时错误（OOM/网络）下个任务重试。`0`=完全不缓存 |
| `CHUNK_MINUTES` | `15` | 单文件最长时长，超过即切片 |
| `CHUNK_SNAP_WINDOW` | `20` | 切点在 ±该秒数内吸附到静音中点；`0`=严格等分 |
| `SILENCE_NOISE_DB` / `SILENCE_MIN_SEC` | `-35` / `0.6` | `silencedetect` 的静音阈值 |
| `MERGE_GAP_SEC` / `MAX_MERGED_SEC` | `0.8` / `60` | 同说话人合并阈值与单段上限 |
| `MERGE_GAP_SEC_NO_DIAR` | `0.3` | 无说话人标签时的合并窗口 |
| `HALLU_NO_SPEECH_PROB` / `HALLU_AVG_LOGPROB` / `HALLU_COMPRESSION_RATIO` | `0.6` / `-1.0` / `2.4` | Whisper 幻觉过滤阈值（BGM/静音处的重复与「感谢观看」类输出） |

### 总结

| 变量 | 默认 | 说明 |
|---|---|---|
| `AUTO_SUMMARIZE` | `false` | 是否在转录完成后自动总结。`false` 时任务停在 `transcribed` |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | 可指向中转/代理 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 换 `deepseek-reasoner` 会自动剔除不支持的参数 |
| `SUMMARY_BLOCK_CHARS` | `8000` | 局部摘要的块大小，决定 Map 调用次数 |
| `AGG_MAX_CHARS` | `24000` | 聚合输入上限，超出先分层压缩 |
| `SUMMARY_MAX_TOKENS` / `FINAL_MAX_TOKENS` | `4096` / `8192` | 后者为 deepseek-chat 输出上限 |
| `SUMMARY_TEMPERATURE` | `0.3` | reasoner 模型下自动忽略 |

---

## 性能与显存

转录耗时与显存峰值取决于模型、量化方式、batch 和显卡。先按显存档位选配置，再用日志里的实际峰值微调。

### 显存档位推荐

| 显存（`nvidia-smi`） | `ASR_MODEL` | `ASR_COMPUTE_TYPE` | `ASR_BATCH_SIZE` | 峰值参考 | 说明 |
|---|---|---|---|---|---|
| 4–6GB | `medium` | `int8_float16` | 2 | 3.5~4.5GB | 分离吃紧时设 `ENABLE_DIARIZATION=false` |
| 6–8GB | `large-v3-turbo` | `int8_float16` | 4 | 4~5GB | 主流显卡，速度与精度平衡 |
| 8–12GB | `large-v3-turbo` | `float16` | 4（可试 8） | 5~6GB | 默认推荐组合 |
| 12–16GB | `large-v3` | `float16` | 4（可试 8） | 6.5~8GB | 精度优先 |
| 16–24GB | `large-v3` | `float16` | 8（可试 16） | 8~11GB | 长视频、批量任务 |
| ≥24GB | `large-v3` | `float16` | 16+ | 10~13GB | 余量充足 |

- 峰值 ≈ Whisper 权重 + 词级对齐模型（约 1.2GB）+ 推理激活 + CUDA 上下文；说话人分离先跑并释放，不与 ASR 叠加。
- `ASR_BATCH_SIZE` 是起始值，OOM 后服务会自动减半重试；稳定后可逐级上调。
- 4~6GB 档位若必须开启说话人分离，可先设 `ASR_ALIGN=0` 省约 1.2GB；仍 OOM 再关 `ENABLE_DIARIZATION`。
- 纯 CPU 可运行（`ASR_DEVICE=cpu` + `int8`），但转录与分离会慢一个数量级。

---

## 常见问题

<details>
<summary><b>Pyannote 加载报 403 / gated repo</b></summary>

用**同一个** HF 账号把这三个模型的条款都点一遍 Agree（缺一个就 403，日志里的 URL 会直接告诉你缺哪个）：

| 模型 | 说明 |
|---|---|
| <https://hf.co/pyannote/speaker-diarization-3.1> | 主管线 |
| <https://hf.co/pyannote/segmentation-3.0> | 分割模型 |
| <https://hf.co/pyannote/speaker-diarization-community-1> | `pyannote.audio` 4.x 的 PLDA 依赖（最容易漏） |

然后生成 Read 权限的 token 填进 `HF_TOKEN`，**重启服务**。进程内会缓存「分离不可用」状态，不重启的话即使补了授权也要等下次启动才会重试。
</details>

<details>
<summary><b>CUDA out of memory</b></summary>

服务端已内置「OOM 自动减半 `batch_size`，最多重试 2 次」。仍失败时按 [显存档位推荐](#显存档位推荐) 降一档：`ASR_BATCH_SIZE` 减半 → `ASR_COMPUTE_TYPE=int8_float16` → `ASR_MODEL` 降档 → `ASR_ALIGN=0` → `ENABLE_DIARIZATION=false`。
</details>

<details>
<summary><b>模型下载慢 / 每次重启都重新下</b></summary>

模型默认缓存到 `%USERPROFILE%\.cache\huggingface`，正常只需下载一次。想换盘就在 `.env` 里设 `HF_HOME=D:/models/huggingface`。若每次启动都在重新下载，检查 `HF_HOME` 是否被改成了临时目录，或临时把 `ASR_OFFLINE=1` 验证本地缓存是否完整。
</details>

<details>
<summary><b>说话人全是 UNKNOWN</b></summary>

说明分离环节降级了（日志里有 `说话人分离失败`）。先看 `GET /healthz`：

- `diarization.last_error` 会给出最近一次失败原因，`diarization.cached_reason` 非空表示本进程已放弃分离（永久性错误才会被缓存）；
- `media_io.soundfile=false` 说明 waveform 解码不可用；
- `media_io.torchcodec=false` 只在 `DIAR_AUDIO_INPUT=path` 时才重要。

本项目默认走 `DIAR_AUDIO_INPUT=waveform`（自己读 16k wav），不依赖 torchcodec。其余检查项：`HF_TOKEN`、三个 gated 模型的 Agree、`ENABLE_DIARIZATION`。
</details>

<details>
<summary><b>分离结果出现「6 位说话人」/ 有人只说了 2 秒</b></summary>

BGM、音效、多人重叠说话会让 pyannote 3.1 过度聚类。本项目已做两层清洗：

1. 总时长低于 `DIAR_MIN_SPEAKER_SEC`（默认 5s）或占比低于 `DIAR_MIN_SPEAKER_SHARE`（默认 1%）的说话人，会按时间就近并入保留的说话人；
2. 已知人数时直接声明：`DIAR_MIN_SPEAKERS=2` + `DIAR_MAX_SPEAKERS=2`，或上传时带 `speakers=2` 表单字段。

仍不理想时，可换 `DIARIZATION_MODEL=pyannote/speaker-diarization-community-1`（同样需要 Agree）。
</details>

<details>
<summary><b>总结里的时间戳和视频对不上</b></summary>

时间戳来自切片偏移的累加，理论精度到 PCM packet 级（~64ms）。如果看到明显偏差，先看 `{task_id}.transcript.json` 的 `segments[].start` 是否正常——那里的值就是喂给模型的原始时间。
</details>

<details>
<summary><b>韩语/日语片段被识别成英文</b></summary>

`auto` 模式下语言是**逐片**检测的（不是整段一次）。前 3 个「有语音」切片投票，只有某个语言占比 ≥ `ASR_LANG_MIN_SHARE`（默认 0.8）才锁定并固定传给后续切片；多语内容或平票则继续逐片检测，顶层 `language` 写成 `multilingual`，每个切片的判定结果记录在 `transcript.json` 的 `chunk_languages` 里。

如果内容语言已知，直接指定更稳妥：

```ini
ASR_LANGUAGE=ko
```

这样所有切片都不会再做语言检测，也省掉每片的检测开销。
</details>

<details>
<summary><b>转录完成后一直停在 transcribed，没有 .md</b></summary>

这是默认行为。`AUTO_SUMMARIZE=false` 时，转录完成后会停住，等你决定是否生成总结：

- 只要转录文本：直接用 `data/outputs/{task_id}.transcript.json`；
- 要总结：`POST /summarize/{task_id}`，或用 `POST /summarize` 上传 JSON；
- 想全自动：`.env` 里设 `AUTO_SUMMARIZE=true` 并重启服务。
</details>

<details>
<summary><b>上传后立刻失败：<code>ASR 模型加载失败: HTTP Error 504: Gateway Time-out</code></b></summary>

加载模型时的网络请求失败，有两个来源：

| 报错形态 | 来源 | 说明 |
|---|---|---|
| `HTTP Error 504: Gateway Time-out`（urllib 风格） | GitHub / torch.hub | 通常是 silero VAD：日志里紧跟 `Performing voice activity detection using Silero...`。`ASR_OFFLINE` 对它无效 |
| `504 Server Error: Gateway Time-out for url: https://huggingface.co/...` | HuggingFace | whisper 权重或对齐模型下载失败 |

判断方法：看失败前最后一条日志。Silero 相关按下表处理，否则走 HuggingFace。

1. **silero VAD 拉不到 GitHub**
   - 最省事：`.env` 设 `VAD_METHOD=pyannote`（`pyannote/segmentation-3.0` 随说话人分离一起缓存），重启后 `POST /transcribe/{task_id}`；
   - 或恢复 `%USERPROFILE%\.cache\torch\hub\snakers4_silero-vad_*` 缓存（旧版本是 `_master`，新版本可能请求 `_main`，复制目录改名即可）；
   - 或给 `HTTP_PROXY` / `HTTPS_PROXY` 配一个能访问 GitHub 的代理；
   - 当前实现遇到这种情况会自动改用 pyannote VAD 再试一次（日志里会写「改用 pyannote VAD 重试」）。
2. **HuggingFace 不可达**
   - 先 `POST /transcribe/{task_id}`；
   - `ASR_OFFLINE=auto`（默认）会自动切 HF 离线模式重试一次；确认缓存存在（如 `%USERPROFILE%\.cache\huggingface\hub\models--Systran--faster-whisper-large-v3-turbo`）后可设 `ASR_OFFLINE=1`；
   - 需要重新下载时，检查代理/VPN，或设 `HF_ENDPOINT=https://hf-mirror.com`。
</details>

<details>
<summary><b>总结失败（402 余额不足 / 网络错误）后需要重新转录吗</b></summary>

不需要。只要 `{task_id}.transcript.json` 还在，修复问题后重新 `POST /summarize/{task_id}` 即可，只重跑总结，不重新转录。`/status` 里的 `result.summarize_url` 就是这个入口。
</details>

<details>
<summary><b>分离失败时为什么不再下载词级对齐模型</b></summary>

对齐（wav2vec2）的唯一用途是把一个 segment 按说话人拆开。分离失败时所有段落都是 `UNKNOWN`，对齐结果不会被读取，却会按语言触发约 1.2GB 下载。因此只要没有说话人标签就跳过对齐（`ASR_ALIGN` 只控制开关）。word 级时间戳目前不写入 `transcript.json`，需要字幕时可以在这里扩展输出。
</details>

<details>
<summary><b>重启后任务全丢了</b></summary>

服务启动时会做两层恢复：

1. 扫描 `data/outputs/*.transcript.json` 重建任务表：有对应 `.md` 的标为 `completed`，没有的标为 `transcribed`；
2. 扫描 `data/uploads` 里「有文件但没有任务记录」的媒体（保留期内、最近 20 个），恢复成 `failed` 任务，可直接 `POST /transcribe/{task_id}` 重新转录。

正在运行的任务无法断点续跑。
</details>

---

## 已知限制

1. 任务状态为单进程内存态，不支持多 worker / 多副本（启动时会从 `data/outputs` 与 `data/uploads` 恢复历史任务，但运行中的任务无法迁移）。
2. 无鉴权与限流，公网暴露前需自行加网关或 Token 校验。
3. 没有「从转录中断点继续」的接口；上传文件在保留期内可直接重新转录。
4. 切片不做重叠，边界半句依靠「切点吸附静音 + 词级拆段 + 字符级去重」修复，极端情况下仍可能丢掉一个词。
5. 同一段 30s 内两人抢话时，说话人切分会有抖动（已做孤词平滑，但不保证完全准确）。
6. 说话人分离失败（授权缺失、显存不足等）时会自动降级为无标签转录。总结按三态处理：有标签走多人 prompt；做了分离但失败走「多人对话未分离」prompt；只有明确没请求分离时才按单人口播处理。


