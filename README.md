# Polyglot Media Summarizer

> 上传一段多语言视频/音频，得到一份带时间戳与说话人归属的中文结构化总结。
> **ASR 全程本地推理**——跑在云端 GPU 容器里，或你自己的 Windows/Linux 机器上；音频不出本地，只有「提炼后的文本」会送给 DeepSeek。

支持 `.mp4 / .mkv / .mov / .webm / .mp3 / .wav / .m4a / .flac ...`，以英文为主、夹杂韩语/日语的音视频内容效果最佳。

---

## 目录

- [核心特性](#核心特性)
- [架构](#架构)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [API 参考](#api-参考)
- [任务状态与进度](#任务状态与进度)
- [产物与文件布局](#产物与文件布局)
- [命令行工具](#命令行工具)
- [配置项](#配置项)
- [性能与显存](#性能与显存)
- [常见问题](#常见问题)
- [隐私与安全](#隐私与安全)
- [已知限制](#已知限制)

---

## 核心特性

| 能力 | 实现 | 说明 |
|---|---|---|
| 音频抽取 | FFmpeg | 任意容器 → 16kHz / 单声道 / PCM16 WAV，`-c:a copy` 切片不重编码 |
| 语音识别 | `murmurai-core`（WhisperX fork）+ `faster-whisper` | 默认 `large-v3`，支持 `large-v3-turbo`；词级时间戳 |
| 说话人分离 | `pyannote.audio` + `pyannote/speaker-diarization-3.1` | 整段只跑一次，跨切片说话人标签一致 |
| 长音频处理 | FFmpeg 按时长切片 + Silero VAD | 超过 15 分钟自动切分，规避显存溢出 |
| 边界拼接 | 词级拆段 + 字符级重叠去重 | 切片切点造成的半句/重复词自动修复 |
| 结构化总结 | DeepSeek（OpenAI SDK 兼容） | 分段摘要 → 全局聚合，输出六段式中文 Markdown |
| 服务化 | FastAPI + BackgroundTasks | 上传即返回 `task_id`，轮询进度 |
| 运行环境 | 云原生（CNB / Docker）与本地（Windows）双路径 | 云端 `cnb:arch:amd64:gpu`（16 核 / 48GB）；本地 8GB 显卡可用（`medium` + `int8_float16`） |
| 配置方式 | 环境变量 + `.env` | 云端由平台注入；本地用项目根目录的 `.env`（`python-dotenv`） |

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
                    │                → [AudioChunk(start_sec, ...)]  │
                    ├────────────────────────────────────────────────┤
                    │ ② transcribe.py                                │
                    │    pyannote 3.1 整段分离 → 全局 speaker 时间片  │
                    │    murmurai 逐片转录 → 对齐 → 词级拆段          │
                    │    时间戳加回全局偏移 → 边界合并去重            │
                    │    → [{start, end, speaker, text}]             │
                    ├────────────────────────────────────────────────┤
                    │ ③ summarizer.py                                │
                    │    Map：按字数分块 → JSON 局部摘要（校验时间戳）│
                    │    Reduce：片段索引 → 六段式中文 Markdown       │
                    └───────────────────┬────────────────────────────┘
                                        │
                     data/outputs/{task_id}.md  +  .transcript.json
```

**为什么分两段？** ASR 只负责「听清 + 分清谁在说」，产出机器可校验的 JSON；语言理解交给 DeepSeek。好处是音频可以彻底本地化，也便于人工校对转录原文。

---

## 目录结构

```
.
├── .cnb.yml                     # 云原生开发环境 + GPU 节点声明
├── Dockerfile                   # CUDA + torch2.9 + ffmpeg + code-server
├── requirements.txt             # Python 依赖（torch 由基础镜像提供，Windows 需手动装）
├── .env.example                 # 本地环境变量模板（复制为 .env，已 gitignore）
├── app/
│   ├── config.py                # 环境变量集中读取（Settings）
│   ├── main.py                  # FastAPI 接口 + 后台任务编排
│   └── services/
│       ├── audio.py             # FFmpeg 抽取与切片
│       ├── transcribe.py        # 转录 + 说话人分离 + 边界合并
│       └── summarizer.py        # DeepSeek 两阶段总结
└── data/                        # 运行时生成（已 gitignore）
    ├── uploads/                 # 原始上传文件
    ├── audio/                   # 16k wav / 切片（任务结束自动清理）
    └── outputs/                 # {task_id}.md、{task_id}.transcript.json
```

---

## 快速开始

三条路径，选一条走完即可（代码完全一致，只是运行环境与参数不同）：

| 路径 | 适合场景 | 显存要求 | 对应小节 |
|---|---|---|---|
| **云端 A**：cnb.cool 云原生开发 | 没有本地 GPU、想跑 `large-v3` | 48GB（平台提供） | 第 2 节 |
| **云端 B**：Docker | 自有 GPU 服务器 / 团队统一环境 | ≥16GB | 第 3 节 |
| **本地**：Windows 直装 | 一台 Windows 工作站（如 RTX 4070 8G） | ≥6GB | 第 4 节 |

### 0. 前置条件

| 项 | 云端（CNB / Docker） | Windows 本地 |
|---|---|---|
| GPU | NVIDIA，≥16GB（`large-v3` + pyannote 峰值约 10~14GB） | RTX 4070 8GB 可用，但模型要降到 `medium` + `int8_float16` |
| 运行时 | CUDA 12.8 基础镜像 / 宿主机装 NVIDIA Container Toolkit | CUDA 12.8 驱动 + **Python 3.10~3.13** + FFmpeg |
| 账号 | [Hugging Face](https://huggingface.co) Token + [DeepSeek](https://platform.deepseek.com) API Key | 同左（写进 `.env`） |
| HF 许可 | 用同一账号接受 <https://hf.co/pyannote/speaker-diarization-3.1> 的条款，否则 403 | 同左 |

### 1. 配置密钥

**云端**：不要写进任何入库文件。

- CNB：个人设置 → 云原生开发 → 个人环境变量（或密钥仓库 `imports`，见 `.cnb.yml` 末尾注释）
- Docker：`docker run -e HF_TOKEN -e DEEPSEEK_API_KEY ...`，或用 `--env-file .env`

**Windows 本地**：

```powershell
copy .env.example .env      # 然后填入两个 token
notepad .env
```

`app/config.py` 启动时读取**项目根目录**的 `.env`（UTF-8 编码，`override=False`——系统环境变量优先于 `.env`）。文件不存在也不报错，容器/云端无需此文件。

### 2. 云端部署 A：cnb.cool 云原生开发（推荐）

> 面向「没有本地 GPU」或「想用 48GB 显存跑 `large-v3`」。

```bash
git push            # 推到 CNB 仓库
```

在仓库页点「云原生开发」启动，`.cnb.yml` 已声明 GPU 节点：

```yaml
$:
  vscode:
    - runner:
        tags: cnb:arch:amd64:gpu      # 16 核 / 48GB 显存
      docker:
        build: Dockerfile
```

启动后会自动跑一次自检（`nvidia-smi` / `ffmpeg -version` / `import murmurai`），失败不阻塞开发环境。

服务跑在容器里，用 CNB 的「业务端口预览」打开 `8000` 即可访问（本机 `127.0.0.1` 在浏览器里不可达）：

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. 云端部署 B：Docker（Linux + NVIDIA Container Toolkit）

> 面向自有 GPU 服务器，或团队想统一环境。**Windows 上也可以跑 Docker Desktop，但调试体验不如直接本地安装（见下一节）。**

```bash
docker build -t pms:dev .
docker run --gpus all -it --rm \
  -e HF_TOKEN -e DEEPSEEK_API_KEY \
  -p 8000:8000 \
  -v "$PWD/data:/workspace/data" \
  pms:dev bash
```

镜像约 6~8GB，首次启动会下载 `large-v3`（~3GB）与 pyannote 权重，落到 `HF_HOME`（容器内默认 `/workspace/.cache/huggingface`，已挂载到宿主 `./data` 之外的独立目录，建议再挂一个卷持久化）。

### 4. 本地运行（Windows，RTX 4070 8G 示例）

不依赖 Docker / WSL，直接用 Windows 上的 Python 跑。**8G 显存必须按下面的参数来，`large-v3` 会直接 OOM。**

#### 4.1 安装 FFmpeg

```powershell
winget install Gyan.FFmpeg     # 装完重开终端
ffmpeg -version                # 有版本输出即成功
```

装在非 PATH 目录也可以，在 `.env` 里写 `FFMPEG_BIN=C:/ffmpeg/bin/ffmpeg.exe`。代码的解析顺序是「环境变量 → PATH → winget / scoop / chocolatey 常见目录 → 报错」，不会依赖任何写死的路径。

#### 4.2 创建虚拟环境并安装依赖

```powershell
py -3.12 --version             # 需要 3.10 ~ 3.13（murmurai-core 不支持 3.14）
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1   # 若提示执行策略错误：Set-ExecutionPolicy -Scope Process Bypass

pip install -r requirements.txt

# CUDA 版 torch 必须单独装：requirements.txt 里刻意不写，避免云端把镜像自带的 CUDA 版覆盖成 CPU 版
pip install torch==2.9.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望输出：2.9.0+cu128 True
# 若是 False，说明装成了 CPU 版，重跑上面那条 pip install
```

#### 4.3 处理 cuDNN 9 DLL（`faster-whisper` 的隐藏坑）

ctranslate2 找不到 cuDNN 9 的 DLL 时会**静默退回 CPU**（日志里看起来一切正常，只是慢十倍）：

```powershell
pip install "nvidia-cudnn-cu12==9.*"
$cudnn = "$PWD\.venv\Lib\site-packages\nvidia\cudnn\bin"
$env:PATH = "$cudnn;$env:PATH"        # 仅当前会话生效
```

#### 4.4 配置 `.env`

```powershell
copy .env.example .env
notepad .env
```

必须填的只有两个：`HF_TOKEN`、`DEEPSEEK_API_KEY`。8G 显存相关的建议值：

| 变量 | 取值 | 原因 |
|---|---|---|
| `ASR_MODEL` | `medium` | `large-v3` fp16 约需 10GB+；`medium` 约 5GB |
| `ASR_COMPUTE_TYPE` | `int8_float16` | 权重 8bit，显存再降约一半，中英混说精度损失很小 |
| `ASR_BATCH_SIZE` | `4` | 服务端遇 OOM 会自动减半重试，但别一开始就 16 |
| `CHUNK_MINUTES` | `10` | 单片更短，峰值显存更低 |
| `DATA_ROOT` | `data` | **相对路径按项目根解析**，不要写 `C:\...` 绝对路径 |

> ⚠️ Windows 上**不要**设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，该选项不受支持（`.cnb.yml` / `Dockerfile` 里那份是给 Linux 容器的）。

#### 4.5 启动与自检

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

另开一个 PowerShell 验证：

```powershell
curl.exe -s http://127.0.0.1:8000/healthz
# {"status":"ok","ffmpeg":true,"ffprobe":true,
#  "config":{"data_root":"D:\\code\\polyglot-media-summarizer\\data",
#            "hf_token_present":true,"deepseek_key_present":true, ...}}

# 先单独验证音频链路（不需要模型，秒级返回，确认 ffmpeg 与中文路径没问题）
python -m app.services.audio .\sample.mp4 --chunk-minutes 0.2
```

首次转录会下载 `medium`（~1.5GB）、对齐模型与 pyannote 权重，默认落在 `%USERPROFILE%\.cache\huggingface`；想换盘就在 `.env` 里设绝对路径 `HF_HOME=D:/models/huggingface`。

> - 用 `curl.exe` 而不是 `curl`：PowerShell 里 `curl` 是 `Invoke-WebRequest` 的别名，不支持 `-F`。
> - 不要加 `--workers`：任务状态存在进程内存里。

### 5. 启动服务（云端 / 本地通用）

```bash
# 云端容器（CNB / Docker）
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Windows 本地（只给本机用，别监听 0.0.0.0）
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# 开发时可加 --reload
```

自检：

```bash
curl -s http://127.0.0.1:8000/healthz | python -m json.tool
# {"status":"ok","ffmpeg":true,"ffprobe":true,
#  "config":{"hf_token_present":true,"deepseek_key_present":true, ...}}
```

> ⚠️ 任务状态存在进程内存里，**只能单 worker 运行**。不要加 `--workers 4`。

### 6. 跑通第一次任务

```bash
# 1) 上传（立即返回 202）
curl -s -X POST http://127.0.0.1:8000/upload -F "file=@./sample.mp4"
# {"task_id":"3b5e2934...","status":"queued","status_url":"/status/3b5e2934...","result_url":"/result/3b5e2934..."}

# 2) 轮询状态
curl -s http://127.0.0.1:8000/status/3b5e2934...
# {"status":"transcribing","stage":"语音转录","progress":57.3,"detail":"转录切片 3/7", ...}

# 3) 下载总结
curl -OJ http://127.0.0.1:8000/result/3b5e2934...
```

> Windows 下把 `curl` 换成 `curl.exe`，下载那步可改用 `curl.exe -o summary.md http://127.0.0.1:8000/result/<task_id>`。

---

## API 参考

### `POST /upload`

`multipart/form-data`，字段名固定为 `file`。响应码 `202`。

| 字段 | 类型 | 说明 |
|---|---|---|
| `task_id` | string | 任务 ID（32 位 hex） |
| `status` | string | 固定 `queued` |
| `filename` | string | 原始文件名（已剥离路径） |
| `size_bytes` | int | 实际落盘字节数 |
| `status_url` / `result_url` | string | 后续查询地址 |

错误：`415` 不支持的扩展名、`413` 超过 `MAX_UPLOAD_MB`、`400` 空文件。

### `GET /status/{task_id}`

| 字段 | 说明 |
|---|---|
| `status` | `queued / extracting / diarizing / transcribing / summarizing / completed / failed` |
| `stage` | 中文阶段名，如「说话人分离」 |
| `progress` | 0~100，一位小数 |
| `detail` | 当前细节，如「转录切片 3/7」 |
| `elapsed_sec` | 从开始处理到现在的秒数 |
| `error` | 失败原因（仅 `failed` 有值） |
| `result` | 完成后包含 `summary_path`、`transcript_path`、`num_segments`、`speakers`、`tokens` 等 |

错误：`404` 任务不存在。

### `GET /result/{task_id}`

返回 `text/markdown; charset=utf-8`，`Content-Disposition: attachment; filename="summary-{task_id}.md"`。
错误：`404` 不存在、`409` 未完成、`410` 产物已被保留期清理。

### `GET /healthz`

`ffmpeg`/`ffprobe` 是否可用、脱敏后的配置快照、内存中任务数。**不返回任何密钥。**

---

## 任务状态与进度

| 状态 | 进度区间 | 触发点 |
|---|---|---|
| `queued` | 0% | 已入队，等待 GPU（`MAX_CONCURRENT_TASKS` 串行） |
| `extracting` | 2 → 15% | `audio.analyze()` 抽取 + 切片 |
| `diarizing` | 15 → 35% | 加载 pyannote + 整段说话人分离 |
| `transcribing` | 35 → 75% | 逐片转录 + 对齐 + 说话人指派 |
| `summarizing` | 75 → 98% | DeepSeek Map → Reduce |
| `completed` | 100% | Markdown 已落盘 |
| `failed` | 停在失败点 | `error` 字段带具体原因 |

状态**不可回退**：进入 `completed`/`failed` 后，迟到的进度回调会被忽略。

---

## 产物与文件布局

```
data/
├── uploads/   {task_id}.mp4                       # 原始上传，默认保留（便于重跑/排查）
├── audio/     {task_id}/xxx.16k.wav  + .partNNN.wav   # 任务结束自动删除
└── outputs/   {task_id}.transcript.json           # 转录原文（可直接人工校对）
               {task_id}.md                       # 最终中文总结
```

转录 JSON 结构（也可直接喂给 `summarizer` 单独重跑）：

```json
{
  "language": "en",
  "segments": [{"start": 12.34, "end": 18.9, "speaker": "Speaker 1", "text": "..."}],
  "speakers": ["Speaker 1", "Speaker 2"],
  "num_segments": 128,
  "duration_sec": 3721.5,
  "diarized": true,
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
python -m app.services.transcribe ./sample.mp4 --out ./demo.json

# ④ 只测总结（复用上一步 JSON，省一次 ASR 开销）
python -m app.services.summarizer ./demo.json --out ./summary.md
```

> 密钥不用 `export`：云端已由平台注入，本地写进 `.env` 即可（`app/config.py` 会自动加载）。
> **Windows** 下把 `./sample.mp4` 换成 `.\sample.mp4`；需要临时设变量时用 `$env:HF_TOKEN="hf_xxx"`。

输出示例：

```
[   12.34 -    18.90] Speaker 1  So the first thing we need to check is the build pipeline
[   19.10 -    24.55] Speaker 2  Right, and OverlayFS was the bottleneck last time
```

---

## 配置项

全部通过环境变量注入，`app/config.py` 是唯一读取入口。本地开发还可以把它们写进**项目根目录的 `.env`**（模板见 `.env.example`）；加载规则是系统环境变量优先、`.env` 兜底，文件不存在也不报错。

路径类配置（`DATA_ROOT` / `UPLOAD_DIR` / `AUDIO_DIR` / `OUTPUT_DIR`）支持三种写法：相对路径（**按项目根解析，与启动目录无关**）、绝对路径（`D:/data`、`/srv/data`）、带变量（`%USERPROFILE%/data`、`$HOME/data`）。

### 必填

| 变量 | 说明 |
|---|---|
| `HF_TOKEN` | Hugging Face Token，用于加载 pyannote（也接受 `HUGGINGFACE_TOKEN` / `HF_HUB_TOKEN`） |
| `DEEPSEEK_API_KEY` | DeepSeek API Key；缺失时任务会在 `summarizing` 阶段失败 |

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
| `ASR_MODEL` | `large-v3` | 可换 `large-v3-turbo`（更快、略降精度） |
| `ASR_LANGUAGE` | `auto` | 英文夹韩日建议 `auto`；强制单语设 `en` |
| `ASR_COMPUTE_TYPE` | `float16` | CPU 下自动降级 `int8` |
| `ASR_BATCH_SIZE` | `16` | OOM 时自动减半重试，最多 2 次 |
| `VAD_METHOD` | `silero` | 免额外许可；`pyannote` 需接受 `segmentation-3.0` 条款 |
| `ASR_ALIGN` | `1` | 词级对齐开关，关闭可省一次模型下载 |
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization-3.1` | 保持 3.1 |
| `ENABLE_DIARIZATION` | `true` | 关掉则只转录不打说话人标签（可省 HF_TOKEN） |
| `DIAR_MIN_SPEAKERS` / `DIAR_MAX_SPEAKERS` | 空 | 已知人数时设置，显著提升准确率 |
| `CHUNK_MINUTES` | `15` | 单文件最长时长，超过即切片 |
| `MERGE_GAP_SEC` / `MAX_MERGED_SEC` | `0.8` / `60` | 同说话人合并阈值与单段上限 |

### 总结

| 变量 | 默认 | 说明 |
|---|---|---|
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | 可指向中转/代理 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 换 `deepseek-reasoner` 会自动剔除不支持的参数 |
| `SUMMARY_BLOCK_CHARS` | `8000` | 局部摘要的块大小，决定 Map 调用次数 |
| `AGG_MAX_CHARS` | `24000` | 聚合输入上限，超出先分层压缩 |
| `SUMMARY_MAX_TOKENS` / `FINAL_MAX_TOKENS` | `4096` / `8192` | 后者为 deepseek-chat 输出上限 |
| `SUMMARY_TEMPERATURE` | `0.3` | reasoner 模型下自动忽略 |

---

## 性能与显存

### 云端（48GB 节点，`large-v3` + `batch_size=16` + pyannote 3.1）

| 内容时长 | 切片数 | 分离 | 转录 | 总结 | 端到端 |
|---|---|---|---|---|---|
| 10 分钟 | 1 | ~30s | ~1.5min | ~30s | ~3 分钟 |
| 1 小时 | 4 | ~3min | ~9min | ~1.5min | ~14 分钟 |
| 3 小时 | 12 | ~9min | ~27min | ~4min | ~40 分钟 |

- 显存峰值 ≈ 10~14GB（faster-whisper `large-v3` + 对齐模型 + pyannote），48GB 节点余量充足。
- 主要成本在转录；想更快可 `ASR_MODEL=large-v3-turbo` + `ASR_BATCH_SIZE=24`。

### 本地（8GB 显存，如 RTX 4070，`medium` + `int8_float16` + `batch_size=4`）

| 内容时长 | 显存峰值 | 端到端（估算） |
|---|---|---|
| 10 分钟 | ~4GB | ~4~6 分钟 |
| 1 小时 | ~5~6GB | ~20~30 分钟 |
| 3 小时 | ~5~6GB | ~60~90 分钟 |

- 与云端的主要差距在转录吞吐，不在分离；`medium` 比 `large-v3` 快 2~3 倍，精度下降主要体现在专业术语与韩/日文片段。
- 显存吃紧时按优先级依次降：`ASR_BATCH_SIZE=2` → `ASR_MODEL=small` → `ENABLE_DIARIZATION=false`（省掉 pyannote 的约 2GB）。
- 说话人分离是在**整段 wav** 上跑的，与切片数无关；长音频分离阶段的显存占用不会因为 `CHUNK_MINUTES` 变小而下降。

### 成本

- 总结成本由 `SUMMARY_BLOCK_CHARS` 决定，1 小时内容约 4~8 次 Map + 1 次 Reduce；每次调用的 token 数都会写进 `/status` 的 `result.tokens`。

---

## 常见问题

<details>
<summary><b>Pyannote 加载报 403 / gated repo</b></summary>

用**同一个** HF 账号：① 在 <https://hf.co/pyannote/speaker-diarization-3.1> 点接受条款；② 生成 Token 并在容器内 `echo $HF_TOKEN` 确认已注入。`VAD_METHOD=pyannote` 时还需要接受 `pyannote/segmentation-3.0`。
</details>

<details>
<summary><b>CUDA out of memory</b></summary>

服务端已内置「OOM 自动减半 `batch_size`，最多重试 2 次」。仍失败时按顺序降级：`ASR_BATCH_SIZE` → `CHUNK_MINUTES` → `ASR_MODEL`（8G 卡请直接用 `medium` + `int8_float16`）。也可临时 `ENABLE_DIARIZATION=false` 只跑转录，省掉 pyannote 的约 2GB。
</details>

<details>
<summary><b>Windows：<code>torch.cuda.is_available()</code> 返回 False</b></summary>

装成了 CPU 版 torch。重装 CUDA 版：

```powershell
pip uninstall -y torch torchaudio
pip install torch==2.9.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128
```

另外确认显卡驱动支持 CUDA 12.8（`nvidia-smi` 右上角的 `CUDA Version` ≥ 12.8）。
</details>

<details>
<summary><b>Windows：转录极慢、GPU 占用接近 0</b></summary>

ctranslate2 找不到 cuDNN 9 的 DLL，静默退回 CPU 了。装 `nvidia-cudnn-cu12==9.*`，并把 `.venv\Lib\site-packages\nvidia\cudnn\bin` 加进 `PATH`（详见[4.3](#43-处理-cudnn-9-dllfaster-whisper-的隐藏坑)）。
</details>

<details>
<summary><b>Windows：提示找不到 ffmpeg</b></summary>

`ffmpeg -version` 能跑但服务找不到，通常是安装后没重开终端（PATH 没刷新）。也可以在 `.env` 里直接指定：

```dotenv
FFMPEG_BIN=C:/ffmpeg/bin/ffmpeg.exe
FFPROBE_BIN=C:/ffmpeg/bin/ffprobe.exe
```

代码会按「环境变量 → PATH → winget/scoop/chocolatey 常见目录」依次探测，失败时的报错信息里可直接看到该配哪个变量。
</details>

<details>
<summary><b>Windows：日志里的中文变成乱码</b></summary>

控制台代码页不是 UTF-8。启动前执行 `chcp 65001`，或在「终端设置」里把编码改成 UTF-8。这不影响接口返回的 JSON（始终 UTF-8）。
</details>

<details>
<summary><b>模型下载慢 / 每次重启都重新下</b></summary>

确保 `HF_HOME` 指向持久目录（默认 `/workspace/.cache/huggingface`），这样工作区保留期内只需下载一次。也可以把模型烘焙进镜像：

```dockerfile
# 可选：构建期预下载，代价是镜像 +3GB
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', download_root='/workspace/.cache/huggingface')"
```

或在 `.cnb.yml` 的启动 stage 里加一条预热命令（首次约 3 分钟，之后命中缓存）。
</details>

<details>
<summary><b>说话人全是 UNKNOWN</b></summary>

说明分离环节降级了（日志里有 `说话人分离失败`）。检查 `HF_TOKEN`、许可、以及 `ENABLE_DIARIZATION`；`/healthz` 里的 `hf_token_present` 可快速确认。
</details>

<details>
<summary><b>总结里的时间戳和视频对不上</b></summary>

时间戳来自切片偏移的累加，理论上精确到 PCM packet 级（~64ms）。如果你看到明显偏差，先看 `{task_id}.transcript.json` 的 `segments[].start` 是否正常——那里的值就是喂给模型的原始时间。
</details>

<details>
<summary><b>韩语/日语片段被识别成英文</b></summary>

`fast-whisper` 的语言检测是按整段音频给出的，混语内容下会自动偏向主导语言。可先按语言人工切分文件，或保持 `ASR_LANGUAGE=auto` 并接受少量误判（总结阶段会保留原文关键词）。
</details>

<details>
<summary><b>重启后任务全丢了</b></summary>

`/upload` 的任务状态是进程内内存字典，重启即清空（`data/outputs` 里的产物还在）。需要持久化请替换 `app/main.py` 的 `_tasks` 为 Redis，并把 `BackgroundTasks` 换成 Celery/RQ。
</details>

---

## 隐私与安全

- **音频不出环境**：ASR 与说话人分离全部在本地 GPU 上完成，不调用任何第三方 ASR 服务。这里的「本地」指**你自己控制的机器**——云端是你租用的 GPU 容器，Windows 是你的工作站（模型与音频都在本机，断网也能转录，只有首次下载权重需要联网）。
- **只有文本出网**：发往 DeepSeek 的是分段后的转录文本，不含原始音视频。若内容敏感，可把 `DEEPSEEK_BASE_URL` 指向自建兼容网关。
- **密钥只走环境变量 / `.env`**：代码中无任何硬编码 Token；`.cnb.yml` 不写入密钥；`.gitignore` 已排除 `.env`，`.env.example` 里只有占位符。
- **上传防护**：文件名剥离路径（防目录穿越）、扩展名白名单、体积上限、流式落盘。
- **日志**：不打印 Token；错误信息会截断到 2000 字符并回传 `/status`。

---

## 已知限制

1. 任务状态为单进程内存态，不支持多 worker / 多副本。
2. 无鉴权与限流，公网暴露前需自行加网关或 Token 校验。
3. 无断点续传与任务重试接口（原始上传文件默认保留，可据此手工重跑）。
4. 切片不做重叠，边界半句依靠「词级拆段 + 字符级去重」修复，极端情况下仍可能丢掉一个词。
5. 同一段 30s 内两人抢话时，说话人切分会有抖动（已做孤词平滑，但不保证完全准确）。

---

**建议的下一步**：`scripts/smoke_test.sh`（造测试音频 → 上传 → 轮询 → 校验产物）、转录结果人工校对 UI、任务持久化（Redis + Celery）。
