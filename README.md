# Polyglot Media Summarizer

> 上传一段多语言视频/音频，先得到一份带时间戳与说话人归属的**本地转录 JSON**，确认后再生成中文结构化总结。
> **ASR 全程本地推理**——跑在云端 GPU 容器里，或你自己的 Windows/Linux 机器上；音频不出本地，只有「提炼后的文本」会送给 DeepSeek。

**默认是两段式，不会自动花钱**：转录（本地 GPU，免费）→ `transcribed` 暂停 → 你显式调用 `/summarize` → DeepSeek 总结（付费）。详见 [两段式流程](#两段式流程转录免费总结付费)。

支持 `.mp4 / .mkv / .mov / .webm / .mp3 / .wav / .m4a / .flac ...`，以英文为主、夹杂韩语/日语的音视频内容效果最佳。

---

## 目录

- [核心特性](#核心特性)
- [架构](#架构)
- [目录结构](#目录结构)
- [模型授权（必读）](#模型授权必读)
- [两段式流程（转录免费，总结付费）](#两段式流程转录免费总结付费)
- [部署总览](#部署总览)
- [部署 A 云端](#部署-a-云端)
- [部署 B Windows 本地](#部署-b-windows-本地)
- [第一次使用](#第一次使用)
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
| 语音识别 | `murmurai-core`（WhisperX fork）+ `faster-whisper` | 默认 `large-v3`，支持 `large-v3-turbo`；词级时间戳按需启用 |
| 说话人分离 | `pyannote.audio` + `pyannote/speaker-diarization-3.1` | 整段只跑一次，跨切片说话人标签一致 |
| 长音频处理 | FFmpeg 按时长切片 + Silero VAD | 超过 15 分钟自动切分，规避显存溢出 |
| 边界拼接 | 词级拆段 + 字符级重叠去重 | 切片切点造成的半句/重复词自动修复 |
| 结构化总结 | DeepSeek（OpenAI SDK 兼容） | 分段摘要 → 全局聚合，输出六段式中文 Markdown |
| **费用控制** | **两段式（转录 / 总结解耦）** | **转录免费且本地完成，JSON 落盘后停在 `transcribed`；只有显式调用 `/summarize` 才产生 API 费用** |
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
                    ├────────────────────────────────────────────────┤
                    │ ② transcribe.py       （本地 GPU，不花钱）      │
                    │    pyannote 整段分离 → murmurai 逐片转录        │
                    │    → 全局时间戳 → 边界合并去重                  │
                    └───────────────────┬────────────────────────────┘
                                        │
                     data/outputs/{task_id}.transcript.json
                                        │
                         ┌──────────────▼──────────────┐
                         │ status = transcribed（暂停）│
                         │ 不调用 DeepSeek，不产生费用 │
                         └──────────────┬──────────────┘
                                        │ 你显式确认（二选一）
                    ┌───────────────────┴────────────────────┐
                    │ POST /summarize/{task_id}              │
                    │ POST /summarize（上传 transcript.json） │
                    └───────────────────┬────────────────────┘
                                        │
                    ┌───────────────────▼────────────────────────────┐
                    │ ③ summarizer.py      （DeepSeek，按 token 计费）│
                    │    Map：按字数分块 → JSON 局部摘要（校验时间戳）│
                    │    Reduce：片段索引 → 六段式中文 Markdown       │
                    └───────────────────┬────────────────────────────┘
                                        │
                             data/outputs/{task_id}.md
```

**为什么分两段？** ASR 只负责「听清 + 分清谁在说」，产出机器可校验的 JSON；语言理解交给 DeepSeek。更重要的是**成本边界**：转录用你自己的 GPU，边际成本为零；DeepSeek 按 token 计费。把两者拆开，就能先白嫖转录、人工校对 JSON，确认值得再花钱总结。`AUTO_SUMMARIZE=true` 可恢复「转录完立即总结」的旧行为。

---

## 目录结构

```
.
├── .cnb.yml                     # 云原生开发环境 + GPU 节点声明
├── Dockerfile                   # CUDA + torch2.9 + ffmpeg + code-server
├── requirements.txt             # Python 依赖（torch 由基础镜像提供，Windows 需手动装）
├── .env.example                 # 本地环境变量模板（复制为 .env，已 gitignore）
├── run.bat                      # 【日常入口】双击即启动（转调 scripts\start-local.bat）
├── app/
│   ├── config.py                # 环境变量集中读取（Settings）
│   ├── main.py                  # FastAPI 接口 + 后台任务编排
│   └── services/
│       ├── audio.py             # FFmpeg 抽取与切片
│       ├── transcribe.py        # 转录 + 说话人分离 + 边界合并
│       └── summarizer.py        # DeepSeek 两阶段总结
├── scripts/
│   ├── check-env.bat            # 【cmd / 双击入口】环境自检，转调 check-env.ps1
│   ├── start-local.bat          # 【cmd / 双击入口】一键启动，转调 start-local.ps1
│   ├── check-env.ps1            # 自检的实际逻辑（B5/B6 的验证步骤打包）
│   └── start-local.ps1          # 启动的实际逻辑（venv + 端口检查 + 开浏览器）
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
| 3 | <https://hf.co/pyannote/speaker-diarization-community-1> | `pyannote.audio` 4.x 会额外请求它的 PLDA 文件（`plda/xvec_transform.npz`） | 403，同上（**最容易漏**） |

然后到 <https://hf.co/settings/tokens> 创建 **Read** 权限的 token（`hf_...`），填进 `.env` 的 `HF_TOKEN`。

三个常见坑：

- 三个页面必须用**同一个账号**点 Agree，token 也必须是这个账号的；
- 点完 Agree 后**重启服务**：进程内会缓存「分离不可用」状态，避免每个任务都重新下载权重再失败；
- 企业/组织账号可能需要管理员审批，`403 ... not in the authorized list` 通常就是这个原因。

> 日志里出现 `Cannot access gated repo for url ...` 时，直接看 URL 里的仓库名，回上表确认是否漏点。

### 1. 不需要 Agree、但会自动下载的模型

| 模型 | 何时下载 | 大小 |
|---|---|---|
| `Systran/faster-whisper-<ASR_MODEL>` | 首次转录 | medium ≈ 1.5GB，large-v3 ≈ 3GB |
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
       POST /summarize  -F file=@x.json   导入别的机器产出的 JSON
   → summarizing → completed
   → data/outputs/{task_id}.md
```

| 场景 | 推荐操作 |
|---|---|
| 只想要转录文本 / 字幕 | 停在 `transcribed`，直接读 `.transcript.json`，不花钱 |
| 确认内容值得总结 | `POST /summarize/{task_id}` |
| 换机器、或复用历史转录 | `POST /summarize` 上传 JSON |
| 想让服务全自动 | `.env` 里设 `AUTO_SUMMARIZE=true` |
| 总结失败（402 / 网络） | 修复后再次 `POST /summarize/{task_id}`，不会重新转录 |

命令行示例（Windows cmd）：

```bat
rem 1) 上传，记下 task_id
curl.exe -s -X POST http://127.0.0.1:8000/upload -F "file=@.\sample.mp4"

rem 2) 等 status 变成 transcribed
curl.exe -s http://127.0.0.1:8000/status/<task_id>

rem 3) 确认要花钱了，再触发总结
curl.exe -s -X POST http://127.0.0.1:8000/summarize/<task_id>

rem 或者：导入一个现成的 transcript.json
curl.exe -s -X POST http://127.0.0.1:8000/summarize -F "file=@.\demo.transcript.json"
```

> `POST /summarize/{task_id}` 只允许在 `transcribed` 或 `failed`（且 JSON 还在）时调用：`queued/extracting/transcribing` 返回 409，`completed` 且 `.md` 已存在也返回 409，避免误触发重复计费。

---

## 部署总览

两条路，二选一，**代码完全一样**，区别只在「在哪台机器上跑」。

| | 部署 A · 云端 | 部署 B · Windows 本地 |
|---|---|---|
| 需要什么 | 一个 CNB 账号 + 浏览器 | 带 NVIDIA 显卡的 Windows 电脑 |
| 显存 | 48GB（平台提供，可跑 `large-v3`） | ≥6GB（RTX 4070 8G 用 `medium`） |
| 要装软件吗 | 不用，环境在云端镜像里 | 要装 Git / Python / FFmpeg |
| 首次耗时 | 约 15~25 分钟（含镜像构建） | 约 40~60 分钟（含模型下载） |
| 数据在哪 | 云端工作区（回收后在保留期内仍在） | 你自己的硬盘 `data/` |
| 适合 | 没显卡、赶时间、要最高精度 | 有显卡、要离线、内容涉密 |

> 全程只需要会两件事：**点 Fork**、**在终端里复制粘贴命令**。下面每一步都写了「怎么做」「成功长什么样」「失败了看哪」。

### 0. 先准备三样东西（两种部署都要）

| # | 东西 | 怎么拿 | 用途 |
|---|---|---|---|
| 1 | GitHub / CNB 账号 | 已有即可 | 存放代码 |
| 2 | `HF_TOKEN` | ① 登录 <https://huggingface.co> → 右上头像 → Settings → Access Tokens → New token，类型选 **Read** → 复制 `hf_...`<br>② 用**同一个账号**给 [3 个 gated 模型](#0-三个必须-agree-的-gated-模型) 逐个点 **Agree**（缺一个就会 403） | 下载说话人分离模型 |
| 3 | `DEEPSEEK_API_KEY` | 登录 <https://platform.deepseek.com> → 左侧 API keys → 创建 → 复制 `sk-...`（只显示一次） | 生成中文总结（**只用转录功能可以不填**） |

> 这两个 token 是**密码级别**的东西：不要发群里、不要提交到仓库。云端怎么注入见 A2~A3，本地怎么放见 B7。

### 1. 名词小抄（看不懂就回来查）

| 名词 | 一句话 |
|---|---|
| 终端 / CMD | 敲命令的窗口。Windows：按 `Win` 键 → 输入 `cmd` → 回车。本文默认 CMD（大多数 Windows 用户的默认终端）；必须用 PowerShell 的地方会单独标注 |
| Fork（CNB 叫「派生」） | 把别人的仓库复制一份到**你自己的账号**下（你才有写权限） |
| clone | 把网上的仓库下载到本地硬盘 |
| 虚拟环境（venv） | 给这个项目单独一套 Python 依赖，不污染其他项目 |
| 环境变量 | 程序启动时读的「外部开关」，本项目用它传 token |
| `.env` | 项目根目录下的文本文件，把环境变量写进去，启动时自动读取（仅本地用） |
| 端口 8000 | 本服务的 HTTP 端口；浏览器打开 `http://127.0.0.1:8000/docs` 就能操作接口 |

---

## 部署 A 云端

> 目标：浏览器里点几下 + 终端里一行命令，得到一个可用的线上服务。**不需要在本地装任何软件。**

### A1. Fork 仓库

1. 打开项目仓库页面（例如 `https://cnb.cool/<原作者>/polyglot-media-summarizer`）。
2. 点右上角 **Fork**（CNB 上叫「派生」）。
3. 归属选**你自己的账号**，仓库名保持默认 → 确认。
4. 完成后你会得到 `https://cnb.cool/<你的账号>/polyglot-media-summarizer`。

> 为什么必须 Fork：云原生开发的配额、密钥、流水线都挂在「你自己的仓库」上；直接在原仓库点启动，你配置的密钥不会生效。

### A2. 建一个「密钥仓库」（只做一次）

云端不方便往仓库里放 `.env`（会被提交入库），标准做法是把 token 放进一个**独立私有仓库**，再让主仓库引用它。

1. 在 CNB 新建仓库，命名 `secret`（名字随意），可见性选 **私有**。
2. 新建文件 `llm.yml`：

```yaml
HF_TOKEN: hf_xxxxxxxxxxxxxxxxxxxx
DEEPSEEK_API_KEY: sk-xxxxxxxxxxxxxxxxxxxx
```

3. 提交。这个仓库**绝对不要设成公开**。

### A3. 把密钥注入开发环境（关键一步）

1. 打开 <https://cnb.cool/profile/workspaces>（个人设置 → 云原生开发）。
2. 找到 **个人环境变量**，按 YAML 数组填写：

```yaml
- imports: https://cnb.cool/<你的账号>/secret/-/blob/main/llm.yml
  use_in_slugs:
    - <你的账号>/polyglot-media-summarizer
```

3. 保存，然后**重启**开发环境才生效。

三个必踩的坑：

| 坑 | 后果 | 正确做法 |
|---|---|---|
| 漏写 `use_in_slugs` | 该项被忽略，静默不注入 | 写全 `<账号>/<仓库名>`，支持 glob |
| `use_in_slugs` 没命中当前仓库 | 同上 | 组织名、仓库名大小写都要对 |
| `imports` 写成外部地址 | 不加载 | 只支持 CNB 站内地址或相对路径 |

> 兜底方案：进入开发环境后临时执行 `export HF_TOKEN=hf_xxx && export DEEPSEEK_API_KEY=sk-xxx`，但**环境重启就失效**，只适合临时试一下。

### A4. 启动开发环境

1. 打开你 Fork 出来的仓库，进入**任意分支页面**（不是设置页）。
2. 点页面**右上角**的「云原生开发」。
3. 选 **WebIDE**（浏览器直接打开，免安装）；想用本地编辑器可选 VS Code / Cursor。
4. 首次启动会按 `Dockerfile` 构建镜像（CUDA + torch + ffmpeg），**约 10~20 分钟**；之后命中缓存，几十秒就绪。

启动后会自动跑一次「环境自检」，终端里能看到 4 段输出，重点看这几行：

```
=== 1/4 GPU ===
NVIDIA L40, 46068 MiB, 550.54.15            ← 有这行说明 GPU 正常
=== 2/4 FFmpeg（路径解析与业务代码一致）===
解析到的 ffmpeg  = /usr/bin/ffmpeg           ← 有具体路径说明 ffmpeg 可用
=== 3/4 Python 依赖 ===
torch 2.9.0+cu128 | cuda True | murmurai ok | fastapi 0.115.x
=== 4/4 配置与数据目录 ===
data_root = /workspace/data
hf_token_present = True                      ← 必须是 True，否则回 A3 检查
deepseek_key_present = True                  ← 同上
```

自检失败不会阻塞环境使用，但上面三行的值必须对。

### A5. 启动服务

在 WebIDE 的终端里（代码目录是 `/workspace`，与 `Dockerfile` 的 `WORKDIR` 一致）：

```bash
cd /workspace
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- 必须监听 `0.0.0.0`，写 `127.0.0.1` 会导致云端端口预览打不开。
- 看到 `Uvicorn running on http://0.0.0.0:8000` 即成功，**这个终端不要关**。

### A6. 从浏览器访问

云端没有「本机 127.0.0.1」，要走 CNB 的端口映射，二选一：

1. WebIDE 控制台的 **PORTS** 面板 → 添加端口 `8000` → 系统生成访问 URL。
2. 或在终端执行：

```bash
echo "$CNB_VSCODE_PROXY_URI" | sed "s/{{port}}/8000/"
# 输出形如 https://xxxx-8000.cnb.run，浏览器打开它
```

在这个地址后面拼 `/docs`，就是可交互的接口文档（上传、查状态、下载都能点）。

### A7. 数据与回收

| 事项 | 说明 |
|---|---|
| 保活 | `.cnb.yml` 里 `keepAliveTimeout: 2h`，2 小时无 IDE/SSH 心跳会被回收 |
| 文件 | 工作区 `/workspace` 在保留期内仍在，`data/outputs/*.md` 不会丢 |
| 任务状态 | 存在**进程内存**，服务重启即清空（产物文件还在） |
| 模型缓存 | `/workspace/.cache/huggingface`，保留期内不用重复下载 |
| 时长上限 | GPU 节点单次最长 18 小时 |

### A8. 云端排错

| 现象 | 原因 | 处理 |
|---|---|---|
| 自检 `hf_token_present = False` | A3 没配对，或改完没重启环境 | 核对 `use_in_slugs` 后重启开发环境 |
| 自检 `cuda False` | 没分到 GPU 节点 | 确认 `.cnb.yml` 里 `runner.tags: cnb:arch:amd64:gpu`；联系组织管理员开通 GPU 配额 |
| 浏览器打不开 8000 | 服务监听了 `127.0.0.1`，或没加端口映射 | 用 `--host 0.0.0.0`，并在 PORTS 面板加 8000 |
| Pyannote 报 403 | 没接受模型条款（常见是漏了 `speaker-diarization-community-1`） | 回 [模型授权（必读）](#模型授权必读)，三个模型都要点 Agree |
| 首次启动卡在构建很久 | 要拉 6~8GB 基础镜像 + 装 torch 系依赖 | 正常现象，等它跑完；后续命中缓存 |
| 构建报 `COPY ... failed to calculate checksum ...: "/requirements.txt": not found` | `.cnb.yml` 里 `build:` 写成了字符串，CNB 的构建上下文因此只有 Dockerfile 自己 | 改成对象写法：`build: { dockerfile: Dockerfile, by: [requirements.txt] }`；Dockerfile 里 COPY 了哪些文件，`by` 就要列全哪些 |
| 打开 IDE 没有 CodeBuddy / 插件全没了、终端只剩 bash | 镜像里装了 code-server → 走「单容器模式」，顶掉了 CNB 默认环境（`cnbcool/default-dev-env` 里预装 CodeBuddy 等） | 在 Dockerfile 里 `code-server --install-extension tencent-cloud.coding-copilot`（本仓库已内置）；或删掉镜像里的 code-server 回到双容器模式 |

### A9. 备选：自有服务器的 Docker

有 GPU 服务器、或想让团队用同一套环境时走这条（可跳过 A1~A6）：

```bash
git clone https://cnb.cool/<你的账号>/polyglot-media-summarizer.git
cd polyglot-media-summarizer
docker build -t pms:dev .
docker run --gpus all -d --name pms \
  -e HF_TOKEN=hf_xxx -e DEEPSEEK_API_KEY=sk-xxx \
  -p 8000:8000 \
  -v "$PWD:/workspace" \
  -v pms-models:/workspace/.cache \
  pms:dev
curl -s http://127.0.0.1:8000/healthz
```

> 镜像里**不** COPY 业务代码（CNB 云原生开发的构建上下文只包含 `.cnb.yml` 里 `build.by` 声明的文件），所以要用 `-v "$PWD:/workspace"` 把仓库挂进容器，否则 `uvicorn app.main:app` 找不到 `app/`。
>
> Dockerfile 用了 BuildKit 的 cache mount 来复用 apt / pip 的下载缓存，所以需要 **Docker 23+**（默认启用 BuildKit），或显式 `DOCKER_BUILDKIT=1 docker build -t pms:dev .`。若你的 Docker 更老、报 `unknown flag: mount`，把 `RUN --mount=...` 那两行参数删掉即可照旧构建，只是失去缓存加速。

前置条件：宿主机装好 NVIDIA 驱动与 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)，并确认下面这条能打印出显卡信息：

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

---

## 部署 B Windows 本地

> 目标：从零在一台 Windows 电脑上跑起来。**按顺序做，每步都有验证命令，过了再下一步。**

### B0. 准备清单

| 项 | 要求 | 怎么确认 |
|---|---|---|
| 系统 | Windows 10 / 11 64 位 | `winver` |
| 显卡 | NVIDIA，显存 ≥ 6GB（RTX 4070 8G 够用） | `nvidia-smi` |
| 驱动 | 支持 CUDA 12.8（`nvidia-smi` 右上角 `CUDA Version` ≥ 12.8） | 低于 12.8 先去官网升级驱动 |
| 磁盘 | ≥ 15GB 空闲（torch 约 3GB + 模型 3~6GB + 数据） | 资源管理器 |
| 软件 | Git、Python（3.10~3.13，**不要 3.14**）、FFmpeg | 见 B1~B3 |

> 全程用 **CMD（命令提示符）**：按 `Win` 键 → 输入 `cmd` → 回车。
> 装完任何软件都要**关掉再重开**窗口，否则 PATH 不生效。
>
> ⚠️ **别在 cmd 里敲 `.ps1`**：cmd 不认识 `.ps1`，会弹出「选择一个应用以打开此 .ps1 文件」。记住对应关系：**cmd 用 `activate.bat` / `scripts\*.bat`，PowerShell 才用 `Activate.ps1` / `scripts\*.ps1`**。下面所有命令都是 cmd 版，直接复制粘贴即可。

### B1. 安装 Git

```bat
winget install --id Git.Git -e --source winget
git --version
```

- 成功标志：`git version 2.x.x`
- 没有 `winget`：去 <https://git-scm.com/download/win> 下载安装包，一路 Next 即可。

### B2. 准备 Python（3.10 ~ 3.13）

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

### B3. 安装 FFmpeg

```bat
winget install --id Gyan.FFmpeg -e --source winget
ffmpeg -version
```

- 成功标志：第一行 `ffmpeg version 7.x`
- 装了但仍提示「不是内部或外部命令」：重开窗口（PATH 才会刷新）；还不行就手动下载 <https://www.gyan.dev/ffmpeg/builds/> 的 `release-essentials` 压缩包，解压到 `C:\ffmpeg`，然后二选一：
  - 把 `C:\ffmpeg\bin` 加进系统 PATH；
  - 或者稍后在 `.env` 里写 `FFMPEG_BIN=C:/ffmpeg/bin/ffmpeg.exe`（更省事，改配置即可）。

### B4. 把仓库克隆到本地

```bat
cd /d "%USERPROFILE%\Documents"
git clone https://cnb.cool/<你的账号>/polyglot-media-summarizer.git
cd polyglot-media-summarizer
dir
```

- 成功标志：`dir` 能看到 `Dockerfile`、`requirements.txt`、`app`。
- 私有仓库会提示输入账号密码：CNB 用访问令牌，GitHub 用 Personal Access Token。
- 小技巧：在文件资源管理器进入该文件夹，地址栏输入 `cmd` 回车，终端就直接开在这个目录。

### B5. 创建虚拟环境并安装依赖

```bat
rem 下面这条的 3.13 换成 B2 里 py -0p 看到的版本（3.10 ~ 3.13）
py -3.13 -m venv .venv

rem cmd 激活脚本是 activate.bat —— 敲 .ps1 会弹「选择一个应用以打开此 .ps1 文件」
.venv\Scripts\activate.bat

rem 必须显示所选版本，且提示符前出现 (.venv)
python -V

python -m pip install --upgrade pip

rem ① 先装 CUDA 版 torch（顺序不能反，原因见下）
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

或者干脆**不激活**：直接指定虚拟环境里的解释器，效果等价，也不会踩任何终端策略：

```bat
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

**为什么 torch 必须放在 requirements 前面？**

- `pip install -r requirements.txt` 会因为 `murmurai-core` 的 `torch>=2.9.0` **间接**装一个 PyPI 默认版 torch（Windows 上那是 CPU 版）；
- 先装 cu128 版，第 ② 步会认为「已满足」，既不重复下载、也不会把 CUDA 版覆盖回 CPU 版；
- 顺序反了（先 ② 后 ①）就会「先装新版、再降级覆盖」，留下 torchvision / torchcodec 与新 torch 不配套，`pip` 报 `torchvision ... requires torch>=x.y`，`pip check` 也会变红。

- 成功标志：`2.9.0+cu128 12.8 True`
- 若是 `2.9.0+cpu None False`：装成 CPU 版了，重跑 ① 那条（别用默认源）。
- 若提示 `torchvision ... requires torch>=2.14.0`：**本项目不使用 torchvision**，执行 `pip uninstall -y torchvision` 即可，不影响任何功能。
- 若提示 `UnicodeDecodeError: 'gbk' codec can't decode byte ...`：`requirements.txt` 被保存成了「中文注释 + 无 BOM」，pip 24.x 会按 GBK 解码。仓库里的版本已改为纯 ASCII，不会出现；若你自己加过中文注释，先 `set PYTHONUTF8=1`（PowerShell 里是 `$env:PYTHONUTF8="1"`）再装，或把文件另存为「UTF-8 with BOM」。

### B6. 让 ctranslate2 找到 cuDNN 9

`torch.cuda.is_available()` 为 True **只说明 PyTorch 能用 GPU**；转录走的是 ctranslate2，它需要单独能找到 cuDNN 9 的 DLL，否则**静默退回 CPU**（不报错，只是慢十倍）。

先判断要不要做这一步：

```bat
python -c "import ctranslate2; print('cuda devices:', ctranslate2.get_cuda_device_count())"
```

- 输出 `>= 1`：已有可用 cuDNN，**直接跳到 B7**。
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

rem 想永久生效：用下面这条打开「环境变量」界面，
rem 在「用户变量」里选中 Path → 编辑 → 新建 → 粘贴绝对路径：
rem     D:\你的路径\polyglot-media-summarizer\.venv\Lib\site-packages\nvidia\cudnn\bin
rundll32 sysdm.cpl,EditEnvironmentVariables

rem 不要用 setx 改 PATH：超过 1024 字符会被截断，还会把系统变量抄进用户变量
```

> PowerShell 等价写法：`$env:PATH = "$PWD\.venv\Lib\site-packages\nvidia\cudnn\bin;$env:PATH"`。
> `set` / `$env:` 都只影响**当前窗口**，关掉就失效——所以之后启动服务的那个窗口要先执行过一次，或者直接用上面的「永久生效」方式。
>
> ✅ **用 `run.bat` 启动就不用管这些**：启动脚本会自动把 `.venv\Lib\site-packages\nvidia\cudnn\bin` 加进本次会话的 PATH（只影响该进程，不写系统变量）。上面这些手工设置只在你**自己手敲 `python -m uvicorn`** 时才需要。

验证（两条都要过）：

```bat
python -c "import ctypes; ctypes.WinDLL('cudnn_ops64_9.dll'); print('cudnn 9 DLL 可加载')"
python -c "import ctranslate2; print('cuda devices:', ctranslate2.get_cuda_device_count())"
```

### B7. 填写 `.env`

```bat
copy .env.example .env
notepad .env
```

| 变量 | 填什么 | 说明 |
|---|---|---|
| `HF_TOKEN` | `hf_...` | 必填；需为 [3 个 gated 模型](#0-三个必须-agree-的-gated-模型) 点过 Agree 的账号 |
| `DEEPSEEK_API_KEY` | `sk-...` | 必填（只有手动触发总结时才会用到） |
| `AUTO_SUMMARIZE` | `false` | 保持 false：转录完先停下，确认后再花钱 |
| `ASR_MODEL` | `medium` | 8G 显存必须降档，`large-v3` 会 OOM |
| `ASR_COMPUTE_TYPE` | `int8_float16` | 进一步省显存 |
| `ASR_BATCH_SIZE` | `4` | 别一上来就 16 |
| `ASR_ALIGN` | `1` | 分离失败时自动跳过词级对齐，省约 1.2GB 下载 |
| `CHUNK_MINUTES` | `10` | 单片更短，峰值显存更低 |
| `DATA_ROOT` | `data` | 相对路径按项目根解析，别写 `C:\...` |

- 12GB 以上显存可以不改这几项，保留 `large-v3` + `float16` + `16`。
- 记事本保存即可（`app/config.py` 按 UTF-8 读取）。
- 路径类配置三种写法都支持：`D:/data`、`%USERPROFILE%/data`、`data`（相对项目根）。

### B8. 启动服务

```bat
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- 成功标志：`Uvicorn running on http://127.0.0.1:8000`
- 启动日志会打印 `data_root` 与 `hf_token_present`，顺手核对路径与密钥。
- 首次转录会下载模型（`medium` 约 1.5GB + pyannote 权重），默认存 `%USERPROFILE%\.cache\huggingface`；想换盘就在 `.env` 里设绝对路径 `HF_HOME=D:/models/huggingface`。
- **不要加 `--workers`**：任务状态在进程内存里。

### B9. 验证

浏览器打开 <http://127.0.0.1:8000/docs>，这是自动生成的接口文档：

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

> CMD 里没有这个坑（`curl` 就是 `curl.exe`）；**PowerShell** 里 `curl` 是 `Invoke-WebRequest` 的别名、不支持 `-F`。所以上面统一写 `curl.exe`，两种终端都能跑。

### B10. 关闭与下次启动

- 关闭：在运行服务的窗口按 `Ctrl + C`（或直接关窗口）。

**下次启动（推荐：一步）**

关机重开后，**不需要重装任何东西**：不用再 `pip install`、不用重建 `.venv`、模型也不会重新下载（都在 `%USERPROFILE%\.cache\huggingface`）。只要：

1. 打开项目文件夹（资源管理器）；
2. 双击 **`run.bat`**。

等价于在 cmd 里：

```bat
cd /d "%USERPROFILE%\Documents\polyglot-media-summarizer"
run.bat
```

脚本会自动完成：定位 `.venv`（不用激活）→ 把 venv 里的 cuDNN 9 加进本次会话的 PATH（**这一条很关键**，否则重启后转录会静默跑在 CPU 上，慢十倍）→ 检查端口 → 启动 uvicorn → 打开 `/docs`。

**手动启动（不想用脚本时）**

```bat
cd /d "%USERPROFILE%\Documents\polyglot-media-summarizer"
.venv\Scripts\activate.bat
set "PATH=%CD%\.venv\Lib\site-packages\nvidia\cudnn\bin;%PATH%"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

> 第 3 行只在「没做永久 PATH」时需要（见 B6）。忘了加也不会报错，只是转录会退回 CPU 变得很慢。

### B11. 本地排错

| 现象 | 原因 | 解决 |
|---|---|---|
| 敲 `Activate.ps1` 弹出「选择一个应用以打开此 .ps1 文件」 | 在 **cmd** 里执行了 PowerShell 脚本，cmd 不认识 `.ps1` | 改用 `.venv\Scripts\activate.bat`；或换到 PowerShell 窗口再执行 `Activate.ps1` |
| `无法加载文件 Activate.ps1，因为在此系统上禁止运行脚本` | PowerShell 执行策略限制（只在 PowerShell 里出现，cmd 不受影响） | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`，或直接用 `activate.bat` |
| `python` / `py` 不是内部或外部命令 | 没勾 Add to PATH，或没重开终端 | 重装 Python 并勾选 PATH，或重开窗口 |
| `No suitable Python runtime found`（`py -3.12` 时） | 机器上没装 3.12；`py` 只挑已装版本，不会自动下载 | `py -0p` 看实际版本，然后用 `py -3.13 -m venv .venv`；确实没有 3.10~3.13 再 `winget install --id Python.Python.3.12 -e` |
| `pip` 报 `UnicodeDecodeError: 'gbk' codec can't decode byte ...` | pip 24.x 在中文 Windows 上按 cp936 解 `requirements.txt`，文件却是无 BOM 的 UTF-8 | 临时：`set PYTHONUTF8=1`（PowerShell 里 `$env:PYTHONUTF8="1"`）后重试；永久：文件存成纯 ASCII 或 UTF-8 with BOM（仓库版本已修正） |
| `pip` 报 `torchvision ... requires torch>=x.y` | 旧环境遗留的 torchvision 与被降级/换版的 torch 不配套 | 本项目不用它：`pip uninstall -y torchvision`，再 `pip check` |
| `torch.cuda.is_available()` 是 False | 装成了 CPU 版 | 按 B5 重装 cu128 版 |
| `ct2 cuda devices: 0` | 缺 cuDNN 9 DLL | 按 B6 处理 |
| 转录极慢、任务管理器里 GPU 占用接近 0 | 已静默退回 CPU：缺少 cuDNN 9 的 DLL（重启后新窗口丢了 PATH 是最常见原因） | 用 `run.bat` 启动会自动注入 PATH；仍为 0 则按 B6 装 `nvidia-cudnn-cu12` 并跑 `scripts\check-env.bat` 第 5 项确认 |
| `未找到可执行文件 'ffmpeg'` | 没进 PATH | 按 B3 设 `FFMPEG_BIN` |
| `OSError: [WinError 1455] 页面文件太小` | 虚拟内存不足（torch + 模型同时加载） | 系统属性 → 高级 → 性能设置 → 高级 → 虚拟内存，调到 16GB 以上 |
| 启动报 `[Errno 10048]` 端口占用 | 8000 被别的程序占了 | 换 `--port 8001` |
| 日志中文乱码 | 控制台代码页不是 UTF-8 | 先执行 `chcp 65001` |
| `CUDA out of memory` | 参数超出显存 | 依次降：`ASR_BATCH_SIZE=2` → `ASR_MODEL=small` → `ENABLE_DIARIZATION=false` |
| 状态一直停在 `transcribed`，没有 `.md` | 默认两段式：转录完成后等你确认，避免自动计费 | 要总结就 `POST /summarize/{task_id}`；想全自动设 `AUTO_SUMMARIZE=true` 并重启 |
| 日志出现 `No active speech found in audio` | 该切片是静音/纯音乐，VAD 没检测到语音 | 正常，跳过该片；若每片都是这行，检查文件是否真的没有音轨或音量过低 |

### B12. 一键脚本（可选，省去手敲命令）

仓库自带两组脚本，把 B5/B6 的验证与启动步骤打包好了。**cmd 用 `.bat`，PowerShell 用 `.ps1`**，功能完全一样：

| 脚本 | 作用 | cmd 用法 | PowerShell 用法 |
|---|---|---|---|
| `run`（项目根目录） | **日常入口**：部署完成后只需双击它，等价于 `scripts\start-local.bat`。命名刻意避开 `start`——那是 cmd 的内建命令，敲 `start` 会开新窗口而不是运行脚本 | `run.bat` | `.\run.bat` |
| `check-env` | 8 项自检：Python 版本 / 依赖 / torch-CUDA / cuDNN / FFmpeg / 密钥 / 数据目录 / 显存与模型档位是否匹配 | `scripts\check-env.bat` | `.\scripts\check-env.ps1` |
| `start-local` | 定位虚拟环境（自动识别 `.venv` / `venv`）→ 检查端口 → 启动 uvicorn → 自动打开 `/docs` | `scripts\start-local.bat` | `.\scripts\start-local.ps1` |

> `.bat` 只是外壳（负责绕过执行策略后再调用同名 `.ps1`），**逻辑只写在 `.ps1` 里**，要改行为请改 `.ps1`。
> 两个 `.bat` 都能在资源管理器里**双击运行**；`check-env.bat` 结束时窗口会停住，方便看结果。

常用参数：

```bat
rem 参数与 PowerShell 版完全一致，只是入口换成 .bat

rem 换端口
scripts\start-local.bat -Port 8001

rem 开发模式，改代码自动重载
scripts\start-local.bat -Reload

rem 不自动开浏览器
scripts\start-local.bat -NoBrowser

rem 首次使用：自动挑 3.10~3.13 建 .venv、装 torch + 依赖（会下载数 GB）
scripts\start-local.bat -Setup

rem 局域网访问（本项目无鉴权，谨慎）
scripts\start-local.bat -BindAddress 0.0.0.0
```

自检输出示例（绿色即通过，红色会给出修复提示）：

```
==== Polyglot Media Summarizer · 环境自检 ====
项目目录: D:\code\polyglot-media-summarizer

1/8 Python 解释器
  [OK]   使用虚拟环境: D:\code\polyglot-media-summarizer\.venv\Scripts\python.exe
4/8 PyTorch 与 CUDA
  [OK]   torch 2.9.0+cu128 · CUDA 可用 · NVIDIA GeForce RTX 4070 · 8.0GB 显存
5/8 cuDNN 9（ctranslate2）
  [OK]   ctranslate2 检测到 1 个 CUDA 设备（cuDNN 9 已就绪）
8/8 配置与数据目录
  [OK]   ASR: model=medium compute=int8_float16 batch=4 chunk=10.0min

全部通过（警告 0 项）。
下一步：scripts\start-local.bat（cmd / 双击；PowerShell 里用 .\scripts\start-local.ps1）
```

> - 需要 PowerShell 5.1+（Windows 10/11 自带）；`.ps1` 保存为 **UTF-8 with BOM**，避免 5.1 下中文乱码。
> - `.bat` 已内置 `-ExecutionPolicy Bypass`，**cmd 用户不需要改任何策略**，直接跑或双击即可。
> - 若在 PowerShell 里直接跑 `.ps1` 被策略拦住，用 `powershell -ExecutionPolicy Bypass -File .\scripts\xxx.ps1` 绕过，不必改全局策略。
> - `check-env.bat` / `check-env.ps1` 有任何 `[FAIL]` 都会以退出码 1 结束；接进 CI 请直接调 `.ps1`（`.bat` 末尾有 `pause`，会等按键）。

---

## 第一次使用

不管用哪种部署，用起来完全一样。流程是：上传 → 抽取音频 → 说话人分离 → 转录 → **停在 `transcribed`** → 你确认 → 总结 → 下载。

### 用浏览器（推荐新手）

1. 打开 `http://127.0.0.1:8000/docs`（云端换成你的端口预览地址 + `/docs`）。
2. `POST /upload` → Try it out → 选择文件 → Execute → 记下 `task_id`。
3. `GET /status/{task_id}` → 反复 Execute，直到 `status` 变成 **`transcribed`**（`progress` 约 75）。
4. 此时 `data/outputs/{task_id}.transcript.json` 已经生成，**没有产生任何 API 费用**。
5. 确认要总结后，执行 `POST /summarize/{task_id}` → Execute → 看 `status` 变成 `completed`。
6. `GET /result/{task_id}` → Execute → 点「Download file」拿到 Markdown。

### 用命令行

```bash
# 1) 上传
curl -s -X POST http://127.0.0.1:8000/upload -F "file=@./sample.mp4"
# {"task_id":"3b5e...","status":"queued","status_url":"/status/3b5e...","result_url":"/result/3b5e..."}

# 2) 轮询到 transcribed —— 此时只有 JSON，没有花一分钱
curl -s http://127.0.0.1:8000/status/3b5e...
# {"status":"transcribed","stage":"转录完成（待总结）","progress":75.0,"detail":"转录完成，等待手动触发总结...", ...}

# 3) 确认内容值得总结，再触发（会调用 DeepSeek）
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

见 [性能与显存](#性能与显存)。粗估：10 分钟内容在 48GB 云端约 3 分钟、在 8G 本地约 5 分钟——这是**转录**耗时。总结另算（几秒到几十秒），而且默认要你手动触发才会执行。

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

> 上传只负责「抽取 + 转录」。默认流程会停在 `transcribed`，**不会自动调用付费 API**。

### `POST /summarize/{task_id}`

对已完成转录的任务触发总结。**会调用 DeepSeek API 并产生费用。**

响应码 `202`，字段含 `task_id`、`status`、`transcript_path`、`status_url`、`result_url`、`note`。

| 场景 | 响应 |
|---|---|
| `transcribed` 或 `failed`（且 JSON 存在） | `202`，开始总结（GPU 忙时排队） |
| `queued` / `extracting` / `diarizing` / `transcribing` | `409` 任务尚未转录完成 |
| `summarizing` | `409` 正在总结，勿重复触发 |
| `completed` 且 `.md` 已存在 | `409` 总结已存在（避免重复计费） |

### `POST /summarize`

导入现成的 `transcript.json` 并总结，适合换机器或复用历史转录。`multipart/form-data`，字段名 `file`，上限 50MB。

接受两种结构：完整转录对象（含 `segments` / `speakers` / `duration_sec` / `diarized`），或直接的 segment 数组。缺 `segments`、文本全空、时间戳无法解析时返回 `400`。

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
错误：`404` 不存在、`409` 未完成（`transcribed` 会明确提示先调用 `/summarize/{task_id}`）、`410` 产物已被保留期清理。

### `GET /healthz`

`ffmpeg`/`ffprobe` 是否可用、脱敏后的配置快照、内存中任务数、等待总结的任务数（`pending_summary`）。**不返回任何密钥。**

---

## 任务状态与进度

| 状态 | 进度区间 | 触发点 |
|---|---|---|
| `queued` | 0% | 已入队，等待 GPU（`MAX_CONCURRENT_TASKS` 串行） |
| `extracting` | 2 → 15% | `audio.analyze()` 抽取 + 切片 |
| `diarizing` | 15 → 35% | 加载 pyannote + 整段说话人分离（失败会降级，不阻断） |
| `transcribing` | 35 → 75% | 逐片转录 + 说话人指派 |
| **`transcribed`** | **75%** | **转录完成、JSON 已落盘，等待你决定是否花钱总结（默认停留点）** |
| `summarizing` | 75 → 98% | 收到 `/summarize` 后调用 DeepSeek Map → Reduce |
| `completed` | 100% | Markdown 已落盘 |
| `failed` | 停在失败点 | `error` 字段带具体原因；若 JSON 还在，`result.retry_url` 可直接重跑总结 |

状态**不可回退**：进入 `completed`/`failed` 后，迟到的进度回调会被忽略；`transcribed` 也会忽略转录阶段的迟到回调，只能通过 `/summarize` 向前推进。

> `transcribed` 不是失败，也不是完成。`/result` 此时返回 409，但 `.transcript.json` 已经可以正常使用。

---

## 产物与文件布局

```
data/
├── uploads/   {task_id}.mp4                       # 原始上传，默认保留（便于重跑/排查）
├── audio/     {task_id}/xxx.16k.wav  + .partNNN.wav   # 任务结束自动删除
└── outputs/   {task_id}.transcript.json           # 转录原文（transcribed 时就有，可直接人工校对）
               {task_id}.md                       # 中文总结（只有 /summarize 成功后才有）
```

`transcript.json` 就是「免费阶段」的完整产物，也可以直接拿去喂给 `POST /summarize` 或命令行工具。

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

> `④` 与 `POST /summarize` 等价：都只读转录 JSON，不会再跑一遍 ASR，也不消耗 GPU 时间。

> 密钥不用 `export`：云端已由平台注入，本地写进 `.env` 即可（`app/config.py` 会自动加载）。
> **Windows** 下把 `./sample.mp4` 换成 `.\sample.mp4`；需要临时设变量时，cmd 用 `set HF_TOKEN=hf_xxx`，PowerShell 用 `$env:HF_TOKEN="hf_xxx"`。

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
| `HF_TOKEN` | Hugging Face Token，用于加载 pyannote（也接受 `HUGGINGFACE_TOKEN` / `HF_HUB_TOKEN`）；需为 [3 个 gated 模型](#0-三个必须-agree-的-gated-模型) 点过 Agree |
| `DEEPSEEK_API_KEY` | DeepSeek API Key；**只在需要总结时必填**，缺失时 `/summarize` 会失败（转录不受影响） |

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
| `ASR_ALIGN` | `1` | 词级对齐开关（`0` 关闭）。对齐只用于「按说话人拆段」，没有说话人标签时一律跳过，避免按语言各下载 1.2GB wav2vec2 |
| `ASR_LANG_VOTE_SAMPLES` | `3` | 前 N 个「有语音」切片参与语言投票 |
| `ASR_LANG_MIN_SHARE` | `0.8` | 某语言占比达到该阈值才锁定全局语言；多语内容/平票继续逐片检测，顶层 `language` 写 `multilingual` |
| `DIARIZATION_MODEL` | `pyannote/speaker-diarization-3.1` | 保持 3.1；需 Agree 的模型见 [模型授权](#模型授权必读) |
| `DIAR_AUDIO_INPUT` | `waveform` | `waveform`=自己解码 16k wav 传入（**不依赖 torchcodec**，推荐）；`path`=把路径交给 pyannote（需 torchcodec 与 torch 版本匹配 + 共享版 FFmpeg） |
| `ENABLE_DIARIZATION` | `true` | 关掉则只转录不打说话人标签（可省 HF_TOKEN） |
| `DIAR_MIN_SPEAKERS` / `DIAR_MAX_SPEAKERS` | 空 | 已知人数时设置，显著提升准确率；`/upload` 的 `speakers` 表单字段等价 |
| `DIAR_TRIM_SILENCE` | `1` | 分离前裁掉首尾静音（长静音开头能省约 25% 分离耗时；整段无语音时直接跳过） |
| `DIAR_TRIM_MIN_SEC` / `DIAR_ENERGY_THRESHOLD` | `1.0` / `0.001` | 裁剪后至少保留的语音时长、能量阈值（≈ -60dBFS） |
| `DIAR_MIN_SPEAKER_SEC` / `DIAR_MIN_SPEAKER_SHARE` | `5` / `0.01` | 低于该时长的「幽灵说话人」并入最近的保留说话人；占比按**有效语音时长**计算，不含静音 |
| `MAX_SPEAKER_ATTACH_GAP` | `2.0` | segment 与最近说话人时间片超过该间隔就标 `UNKNOWN`（宁缺勿错） |
| `DIAR_CACHE_FAILURE` | `1` | 只缓存永久性失败（403/token）；临时错误（OOM/网络）下个任务重试。`0`=完全不缓存 |
| `CHUNK_MINUTES` | `15` | 单文件最长时长，超过即切片 |
| `CHUNK_SNAP_WINDOW` | `20` | 切点在 ±该秒数内吸附到静音中点，避免从句子中间切开；`0`=严格等分 |
| `SILENCE_NOISE_DB` / `SILENCE_MIN_SEC` | `-35` / `0.6` | `silencedetect` 的静音阈值 |
| `MERGE_GAP_SEC` / `MAX_MERGED_SEC` | `0.8` / `60` | 同说话人合并阈值与单段上限 |
| `MERGE_GAP_SEC_NO_DIAR` | `0.3` | 无说话人标签时的合并窗口；比默认更小，避免把不同人的话揉成一段 |
| `HALLU_NO_SPEECH_PROB` / `HALLU_AVG_LOGPROB` / `HALLU_COMPRESSION_RATIO` | `0.6` / `-1.0` / `2.4` | Whisper 幻觉过滤阈值（BGM/静音处的复读、"感谢观看"类输出） |

### 总结

| 变量 | 默认 | 说明 |
|---|---|---|
| `AUTO_SUMMARIZE` | `false` | 是否在转录完成后自动总结。`false` 时任务停在 `transcribed`，需手动 `/summarize`，避免意外计费 |
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
- 上表的端到端时间**不含首次模型下载**：对齐开启且分离成功时，首次处理某语言还会下载约 1.2GB 的 wav2vec2 对齐模型（中文/韩语各一份）。分离失败或 `ASR_ALIGN=0` 时不会触发。

### 成本

- 总结成本由 `SUMMARY_BLOCK_CHARS` 决定，1 小时内容约 4~8 次 Map + 1 次 Reduce；每次调用的 token 数都会写进 `/status` 的 `result.tokens`。

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

服务端已内置「OOM 自动减半 `batch_size`，最多重试 2 次」。仍失败时按顺序降级：`ASR_BATCH_SIZE` → `CHUNK_MINUTES` → `ASR_MODEL`（8G 卡请直接用 `medium` + `int8_float16`）。也可临时 `ENABLE_DIARIZATION=false` 只跑转录，省掉 pyannote 的约 2GB。
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

说明分离环节降级了（日志里有 `说话人分离失败`）。先看 `GET /healthz`：

- `diarization.last_error` 会给出最近一次失败原因，`diarization.cached_reason` 非空表示本进程已放弃分离（永久性错误才会被缓存）；
- `media_io.soundfile=false` 说明 waveform 解码不可用；
- `media_io.torchcodec=false` 只在 `DIAR_AUDIO_INPUT=path` 时才重要。

历史上最常见的是 `torchcodec is not available`——pyannote.audio 4.x 默认用 torchcodec 解码，而它必须与 torch 版本严格匹配、Windows 上还需要共享版 FFmpeg DLL。本项目现在默认走 `DIAR_AUDIO_INPUT=waveform`（自己读 16k wav），不会再触发这个错误。若你手动改成了 `path`，请改回 `waveform`。

其余检查项：`HF_TOKEN`、三个 gated 模型的 Agree、`ENABLE_DIARIZATION`。
</details>

<details>
<summary><b>分离结果出现「6 位说话人」/ 有人只说了 2 秒</b></summary>

BGM、音效、多人重叠说话会让 pyannote 3.1 过度聚类。本项目已做两层清洗：

1. 总时长低于 `DIAR_MIN_SPEAKER_SEC`（默认 5s）或占比低于 `DIAR_MIN_SPEAKER_SHARE`（默认 1%）的说话人，会按时间就近并入保留的说话人；
2. 已知人数时直接声明，比任何后处理都有效：`DIAR_MIN_SPEAKERS=2` + `DIAR_MAX_SPEAKERS=2`，或上传时带 `speakers=2` 表单字段。

仍不理想时，可换 `DIARIZATION_MODEL=pyannote/speaker-diarization-community-1`（同样需要 Agree）。
</details>

<details>
<summary><b>总结里的时间戳和视频对不上</b></summary>

时间戳来自切片偏移的累加，理论上精确到 PCM packet 级（~64ms）。如果你看到明显偏差，先看 `{task_id}.transcript.json` 的 `segments[].start` 是否正常——那里的值就是喂给模型的原始时间。
</details>

<details>
<summary><b>韩语/日语片段被识别成英文</b></summary>

`auto` 模式下语言是**逐片**检测的（不是整段一次），所以长视频里前几片可能被判成 `en`、后面才变成 `ko`。

现在的策略是：前 3 个「有语音」切片投票，**只有某个语言占比 ≥ `ASR_LANG_MIN_SHARE`（默认 0.8）才锁定**并固定传给后续切片；多语内容或平票则继续逐片检测，顶层 `language` 写成 `multilingual`，每个切片的判定结果记录在 `transcript.json` 的 `chunk_languages` 里。

这样处理日语开场 + 英文主体 + 韩语片段的混合内容时，不会再被强行按单一语言解码。

如果内容语言已知，最稳妥的做法是直接指定：

```ini
ASR_LANGUAGE=ko
```

这样所有切片都不会再做语言检测，也省掉每片的检测开销。
</details>

<details>
<summary><b>转录完成后一直停在 transcribed，没有 .md</b></summary>

这是默认行为，不是卡住。`AUTO_SUMMARIZE=false` 时，转录（免费）完成后会停住，等你决定是否花钱总结：

- 只要转录文本：直接用 `data/outputs/{task_id}.transcript.json`；
- 要总结：`POST /summarize/{task_id}`，或用 `POST /summarize` 上传 JSON；
- 想全自动：`.env` 里设 `AUTO_SUMMARIZE=true` 并重启服务。
</details>

<details>
<summary><b>总结失败（402 余额不足 / 网络错误）后需要重新转录吗</b></summary>

不需要。只要 `{task_id}.transcript.json` 还在，修复问题后重新 `POST /summarize/{task_id}` 即可，只重跑总结，不重新上传、不重新转录。`/status` 里的 `result.retry_url` 就是这个入口。
</details>

<details>
<summary><b>分离失败时为什么不再下载词级对齐模型了</b></summary>

对齐（wav2vec2）的唯一用途是把一个 segment 按说话人拆开。分离失败时所有段落都是 `UNKNOWN`，对齐结果不会被读取，却会按语言触发约 1.2GB 下载。因此**只要没有说话人标签就跳过对齐**（`ASR_ALIGN` 只控制开关，不再有「强制」语义）。word 级时间戳目前也不会写入 `transcript.json`，需要字幕时可以在这里扩展输出。
</details>

<details>
<summary><b>重启后任务全丢了</b></summary>

服务启动时会扫描 `data/outputs/*.transcript.json` 重建任务表：有对应 `.md` 的标为 `completed`，没有的标为 `transcribed`，因此重启后 `/summarize/{task_id}`、`/result/{task_id}` 仍然可用。但**运行中的任务**无法恢复，仍需替换 `app/main.py` 的 `_tasks` 为 Redis，并把 `BackgroundTasks` 换成 Celery/RQ。
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

1. 任务状态为单进程内存态，不支持多 worker / 多副本（服务启动时会从 `data/outputs/*.transcript.json` 恢复历史任务，但运行中的任务无法迁移）。
2. 无鉴权与限流，公网暴露前需自行加网关或 Token 校验。
3. 支持从转录 JSON 重跑总结（`POST /summarize/{task_id}`、`POST /summarize`），但没有「从转录中断点继续」的接口；上传文件默认保留，可重新上传。
4. 切片不做重叠，边界半句依靠「切点吸附静音 + 词级拆段 + 字符级去重」修复，极端情况下仍可能丢掉一个词。
5. 同一段 30s 内两人抢话时，说话人切分会有抖动（已做孤词平滑，但不保证完全准确）。
6. 说话人分离失败（授权缺失、显存不足等）时会自动降级为无标签转录。总结按三态处理：有标签走多人 prompt；**做了分离但失败**走「多人对话未分离」prompt（不编造 Speaker、不假设单口播）；只有明确没请求分离时才按单人口播处理。

---

**建议的下一步**：`scripts/smoke_test.sh`（造测试音频 → 上传 → 轮询 → 校验产物）、转录结果人工校对 UI、任务持久化（Redis + Celery）。
