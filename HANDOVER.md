# 交接：多语种识别 + 重叠语音分离

> 写于 2026-11-09。明天继续时先读这一份，按「当前进度 → 下一步」执行。

## 0. 项目与目标

- 项目：`D:\projects\polyglot-media-summarizer`（FastAPI + murmurai/WhisperX + pyannote 4.x + DeepSeek）
- 目标：多语种采访/直播识别，支持片内 ja/en 切换，并尽量解析多人同时说话（含不同语种）。
- 当前主服务环境：`.venv`（pyannote.audio 4.x，正常）
- 分离独立环境：`.venv-sep`（pyannote.audio 3.3.2，正在收尾）

## 1. 今天已经落地的代码（全部在主服务里，可回退）

### 1.1 片内语言细分（P1，已生效）

| 位置 | 内容 |
|---|---|
| `transcribe.py` | `LanguageWindow` / `LanguageSpan` dataclass |
| | `merge_language_windows()`：窗口合并、低置信补齐、超长拆分、碎片保留 |
| | `_language_switch_count()`：抖动判断只看语言切换次数 |
| | `probe_language()` + `_parse_language_result()`：兼容 `str` / tuple / dataclass / dict / 反序 / 0-100 / logprob / `Language.JA` |
| | `plan_language_spans()`：25s 滑窗、静音吸附、单语不拆、抖动回退 |
| | `_read_waveform_span()`：soundfile 优先，缺失时用标准库 `wave` 兜底 |
| | `_cut_subchunk()`：写 16k PCM16 子片，soundfile 缺失时用 `wave` 兜底 |
| 主循环 | 每个子片用自己的语言 `transcribe` + `align`，时间戳加回全局 |
| transcript | 新增 `language_spans`（每段语言/置信度/段数） |

关键参数（`.env` 可覆盖）：

```ini
ASR_LANG_SPAN=1
ASR_LANG_PROBE_SEC=25
ASR_LANG_SPAN_MAX=300
ASR_LANG_SPAN_MIN=8
ASR_LANG_MIN_CONF=0.55
ASR_LANG_SNAP_WINDOW=6
ASR_LANG_MIN_CHUNK_SEC=45
ASR_LANG_MAX_PARTS=6      # 语言切换次数，不是子片数
ASR_LANG_LOCK=0           # 默认不再锁定全局语言
```

### 1.2 重叠语音标注

- `_speakers_for_span()` 保留所有覆盖比例达标的说话人。
- `_apply_speaker_fields()` 给 segment 写 `speaker` / `speakers` / `overlap=true`。
- `postprocess_segments()` 不合并重叠段；`renumber_speakers()` 同时映射 `speakers` 列表。
- `ASR_OVERLAP_MIN_RATIO` 默认从 `0.15` 调到 `0.3`。

### 1.3 粘连文本与幻觉

- `_max_ngram_repeat()` + `_truncate_at_repeat()`：段内复读截断，剩余 <30% 才丢。
- `_glued_latin_cut()`：连续 26+ 无空格拉丁串。
  - 前半正常 → 截断保留前半；
  - 整段粘连 → **保留原文并标 `low_quality=true`**，总结阶段跳过。
- `_join_words()`：相邻单字符拉丁 token 不插空格（修 `H e l l o`）。
- `_split_segment_by_words()`：字符级 token 占比 ≥50% 时放弃拆段，避免二次破坏文本。

### 1.4 重叠语音分离（第三步，代码已写，环境收尾中）

| 文件 | 作用 |
|---|---|
| `app/services/separation.py` | `overlap_spans()` / `available()` / `separate()`（subprocess） |
| `scripts/separate_overlap.py` | 在 `.venv-sep` 里跑 pyannote 3.3.2，输出每说话人 wav + `manifest.json` |
| `scripts/setup-separation-env.bat` | 纯 ASCII 安装脚本（避免 CMD 乱码） |
| `transcribe.py` | `replace_overlaps_with_separated()`：释放 ASR → 分离 → 逐轨 ASR → 替换原段 |
| 主流程 | `SEPARATION_ENABLED=1` 且环境存在时自动启用，失败回退 |

新增配置：

```ini
SEPARATION_ENABLED=1
SEPARATION_PYTHON=D:/projects/polyglot-media-summarizer/.venv-sep/Scripts/python.exe
SEPARATION_MODEL=pyannote/speech-separation-ami-1.0
SEPARATION_TIMEOUT=600
SEPARATION_MIN_OVERLAP_SEC=0.8
SEPARATION_MAX_CLIP_SEC=60
SEPARATION_RELEASE_ASR=1
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # pyannote 3.3.2 + torch 2.6 必要
```

### 1.5 接口与配置

- `POST /upload` 的 `speakers` 改为 `str` + `default=""`，Swagger 里默认空框（不再预填 1/0），服务端手写 1-20 校验。
- 路由顺序：`/upload` → `/status/{id}` → `/summarize/{id}` → `/result/{id}` → `/transcribe/{id}` → `/summarize` → `/healthz`。
- `.env.example` 默认启用 `HF_HOME=D:/models/huggingface`、`TORCH_HOME=D:/models/torch`。

## 2. 当前进度（最重要）

### 2.1 已经在用户机器上验证成功的

- 语言细分生效：670s 片段切成 `ja 0~50s / en 50~250s / ja 250~350s / en 350~600s`，英日不再混排。
- 重叠标注：`overlap_segments` 有值，段上出现 `speakers` / `overlap`。
- `soundfile` 已装，`detect_language` 返回 `str`（已兼容）。

### 2.2 分离环境现状（`.venv-sep`）

已确认版本：

```
torch 2.6.0+cu124   cuda True
torchaudio 2.6.0+cu124
pyannote 3.3.2
transformers 4.57.6
huggingface_hub 0.36.2
```

待确认/待装：

- `speechbrain==1.0.0`（1.1.2 要求 torch>=2.7.1，必须降级）
- `matplotlib`（昨天最后一步报 `No module named 'matplotlib'`）
- 可能还需要 `torchvision==0.21.0+cu124`（目前只是警告）

### 2.3 昨天最后的报错

```
重叠段分离失败（退出码 2）：torchvision is not available - cannot save figures
[separate_overlap] 加载分离模型失败：No module named 'matplotlib'
```

说明 `pipeline = Pipeline.from_pretrained(...)` 在 import 阶段就失败，模型还没开始下载。

## 3. 明天的第一步（照抄执行）

```bat
cd /d D:\projects\polyglot-media-summarizer

REM 1) 补 matplotlib
.venv-sep\Scripts\pip.exe install matplotlib

REM 2) 如果还报 torchvision 相关错误（不只是警告）
.venv-sep\Scripts\pip.exe install "torchvision==0.21.0+cu124" --extra-index-url https://download.pytorch.org/whl/cu124

REM 3) 确认 speechbrain 已降级
.venv-sep\Scripts\pip.exe install "speechbrain==1.0.0"

REM 4) 冒烟测试（40 秒重叠片段，会下载约 1.3GB 模型）
set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
.venv\Scripts\python.exe -c "from app.services import separation; from app.config import settings; m = separation.separate('test_overlap.wav', 0, 40, 'sep_test', settings); print(m)"
```

成功标准：

- `sep_test\track_00.wav`、`track_01.wav` 存在；
- `sep_test\manifest.json` 里 `tracks` 至少有 1 条；
- 听一下两条轨，确认人声是否被分开。

失败对照：

| 报错 | 处理 |
|---|---|
| `401/403` / `gated` | 用同一 HF 账号在两个模型页点 Agree；确认 `.env` 的 `HF_TOKEN` |
| `Weights only load failed` | 确认 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` 已设 |
| `torchaudio has no attribute AudioMetaData` | torch 又被换了，重跑 `--force-reinstall --no-deps` 固定 2.6.0+cu124 |
| `No module named 'transformers.xxx'` | 确认 `transformers 4.57.6`、`huggingface_hub 0.36.2` |
| `没有可用的分离轨` | 换一段更明显的重叠（567~600 或 600~640）再试 |

## 4. 冒烟通过后（第二、三步）

1. `.env` 写入：

```ini
SEPARATION_ENABLED=1
SEPARATION_PYTHON=D:/projects/polyglot-media-summarizer/.venv-sep/Scripts/python.exe
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

2. 重启主服务，启动日志应出现：

```
重叠语音分离已启用（模型 pyannote/speech-separation-ami-1.0，独立环境 ...）
```

3. 重跑 670s 片段，重点看：

```
切片 0 语言探测：...
切片 0 语言细分：0~50s=ja，50~250s=en，250~350s=ja，350~600s=en
重叠段分离：567.9~609.2s（41.3s），模型 ...
重叠段分离完成 X.Xs：N 轨（['SPEAKER_00', ...]）
重叠段分离替换：原 ... 段 -> 保留 ... 段 + 分离 ... 段
```

4. 检查 `transcript.json`：

- `separated_segments` > 0；
- 重叠段带 `separated: true`、`overlap: true`；
- 分离失败的粘连段带 `low_quality: true`。

5. 如果分离质量差：

- 调 `SEPARATION_MIN_OVERLAP_SEC`（0.8 → 1.5，减少误触发）；
- 调 `SEPARATION_MAX_CLIP_SEC`（60 → 30，降低单次显存）；
- 临时关闭：`SEPARATION_ENABLED=0`，主流程自动回退。

## 5. 今天踩过的坑（按顺序，明天别重复）

1. `.bat` 按 UTF-8 保存、CMD 按 GBK 解析 → 中文 `REM` 行被当命令；已改纯 ASCII。
2. Python 3.13 + Windows，cu124 索引没有 torch 2.5.x 的 cp313 wheel → 必须 2.6.0+cu124。
3. 安装 `pyannote.audio[separation]` 时 pip 把 torch 换成 PyPI 上版本号更高的 **2.15.0 CPU 版** → 安装 pyannote 时必须再次钉住 `torch==2.6.0+cu124`。
4. `huggingface_hub 2.4` 移除了 `use_auth_token` → pyannote 3.3.2 加载失败；降 `<1`。
5. `transformers 5.22.2` 太新 → 降 `<5`。
6. `speechbrain 1.1.2` 要求 `torch>=2.7.1` → 降 `==1.0.0`。
7. 缺 `matplotlib` → `Pipeline.from_pretrained` 直接失败 → 补装。

## 6. 关键代码位置（明天直接跳）

- 语言细分：`app/services/transcribe.py`
  - `merge_language_windows` / `_language_switch_count`
  - `probe_language` / `_normalize_language_code` / `_as_probability` / `_parse_language_result`
  - `plan_language_spans` / `_read_waveform_span` / `_cut_subchunk`
  - 主循环 `transcribe_task` 里的 `for span_index, span in enumerate(spans)`
- 重叠标注：`_speakers_for_span` / `_apply_speaker_fields` / `postprocess_segments`
- 粘连处理：`_max_ngram_repeat` / `_truncate_at_repeat` / `_glued_latin_cut` / `_filter_hallucinations`
- 分离：`replace_overlaps_with_separated`（transcribe.py）、`app/services/separation.py`、`scripts/separate_overlap.py`
- 配置：`app/config.py` 的 `SEPARATION_*`，`.env.example` 的分离段

## 7. 验证状态（今天收尾时）

- `tests/test_pipeline_utils.py`：60 项通过；5 项因容器缺真 torch/ffprobe 失败（与改动无关）。
- `read_lints`：0 报错。
- `git diff --check`：干净。
- 用户实测：语言细分成功；重叠段（567~634s）仍粘连，待分离环境跑通后验证。

## 8. 明天开场白（直接发给我）

> 继续做重叠语音分离。环境已装到 matplotlib 这步，`.venv-sep` 里 torch 2.6.0+cu124 / pyannote 3.3.2 正常。先按 HANDOVER.md 跑冒烟测试，结果贴给你。

## 9. 还没做、但值得排期的

- 分离成功后，把 `scripts/separate_overlap.py` 改成「一次子进程处理多个重叠区间」，避免每个区间重复加载模型（现在每个区间 3~10s 启动开销）。
- 分离轨与原始 diarization 的说话人映射：目前直接用 `manifest.tracks[].speaker`，若分离轨的 SPEAKER_00 与全局 SPEAKER_00 不是同一人，需要按时间重叠重新聚类。
- 日语/游戏直播素材上的分离质量评估（模型是 AMI 英语会议训练）。
- 若 AMI 模型效果差，备选：`pyannote/separation-ami-1.0` 底层模型 + 自写滑窗，或换 MossFormer2 / 商用 API。
