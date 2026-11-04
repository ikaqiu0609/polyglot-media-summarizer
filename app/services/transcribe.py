"""转录 + 说话人分离模块（纯本地推理，不调用任何云端 ASR）。

流程：
    1. murmurai（WhisperX fork，`import murmurai` 等价于 `import whisperx`）加载 ASR 模型；
    2. 对**整段 16k wav** 跑一次 Pyannote 3.1 说话人分离 —— 只在整段上跑，
       否则每片独立聚类，SPEAKER_00 在片与片之间根本不指同一个人；
    3. 逐片转录（切片信息来自 audio.py），把结果时间戳加回全局偏移；
    4. 用 word 级时间戳把跨说话人的 segment 拆开，再按说话人合并；
    5. 跨片边界做「同说话人 + 间隙很小」的合并，并去除切点重复文本；
    6. finally 里 del 模型 + torch.cuda.empty_cache() 释放显存。

关于库 API 的不确定性（重要，读代码前先看）：
    - murmurai-core 与 whisperx 的 API 完全一致，但 3.2/3.3 之间参数名有漂移
      （`vad_onset/vad_offset` 已废弃，改用 `chunk_size`；`load_model` 的缓存目录
      参数在版本间可能叫 `download_root` 或 `model_dir`）。为避免因版本差异直接
      TypeError，这里统一用 `_filter_kwargs()` 按目标函数签名过滤参数。
    - pyannote.audio 3.x 用 `use_auth_token=`，4.x 改成 `token=`。本模块按
      `Pipeline.from_pretrained` 的签名自动选择，并对 TypeError 做二次回退，
      两条路都走不通才报错。
    - pyannote.audio 4.x 的 `pipeline(audio)` 返回 `DiarizeOutput`（真正结果在
      `.speaker_diarization`），3.x 直接返回 `Annotation`。这里统一归一化。
"""

from __future__ import annotations

import gc
import inspect
import json
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from app.services.audio import AudioChunk, analyze, cleanup, probe_duration

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置（环境变量优先，均有安全默认值）
# ---------------------------------------------------------------------------
DEVICE = os.getenv("ASR_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
ASR_MODEL = os.getenv("ASR_MODEL", "large-v3")
COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE", "float16")
if DEVICE == "cpu" and COMPUTE_TYPE in {"float16", "int8_float16"}:
    # CPU 不支持 fp16 推理，自动降级，避免 ctranslate2 直接抛错
    COMPUTE_TYPE = "int8"

BATCH_SIZE = int(os.getenv("ASR_BATCH_SIZE", "16"))
# auto = 让 Whisper 自己判定（本项目英文为主、夹韩/日文，强烈建议 auto）；
# 想强制单语时可设 ASR_LANGUAGE=en
LANGUAGE = os.getenv("ASR_LANGUAGE", "auto")
VAD_METHOD = os.getenv("VAD_METHOD", "silero")  # silero（无需额外许可） / pyannote

# 词级对齐三态：
#   auto（默认）—— 只有说话人分离成功时才对齐。对齐结果仅用于「按说话人拆段」，
#                  分离失败时它毫无用处，却会触发 1.2GB/语言的 wav2vec2 下载。
#   1 / true    —— 强制对齐（即使没有说话人标签，也保留 word 级时间戳）。
#   0 / false   —— 完全关闭。
ALIGN_MODE = (os.getenv("ASR_ALIGN", "auto") or "auto").strip().lower()
ALIGN_ENABLED = ALIGN_MODE not in {"0", "false", "no", "off"}
ALIGN_ONLY_WITH_DIARIZATION = ALIGN_MODE in {"auto", ""}

DIARIZATION_MODEL = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
DIAR_MIN_SPEAKERS = os.getenv("DIAR_MIN_SPEAKERS")  # 可选：已知说话人数时提升准确率
DIAR_MAX_SPEAKERS = os.getenv("DIAR_MAX_SPEAKERS")

MERGE_GAP_SEC = float(os.getenv("MERGE_GAP_SEC", "0.8"))       # 同说话人合并阈值
# 没有说话人标签时，所有人都是 UNKNOWN，若沿用 0.8s 会把不同人的话揉进同一段
MERGE_GAP_SEC_NO_DIAR = float(os.getenv("MERGE_GAP_SEC_NO_DIAR", "0.3"))
MAX_MERGED_SEC = float(os.getenv("MAX_MERGED_SEC", "60"))        # 合并后单段上限，防止整集变一段

# 前 N 个「有语音」的切片各投一票决定全局语言，之后固定传给所有切片。
# 逐片 auto 检测会让同一文件的语言来回跳（例如前 20 分钟被判成 en、之后是 ko）。
LANG_VOTE_SAMPLES = max(1, int(os.getenv("ASR_LANG_VOTE_SAMPLES", "3")))

ASR_OPTIONS: dict[str, Any] = {
    # 长音频 + 无重叠切片场景下，关闭上文条件可显著减少「复读机」幻觉
    "condition_on_previous_text": False,
    "word_timestamps": True,  # 说话人逐词对齐需要
    "without_timestamps": False,
}
MODEL_DIR = os.getenv("ASR_MODEL_DIR") or None  # 不设则用 HF_HOME 下的默认缓存


class TranscriptionError(RuntimeError):
    """转录 / 说话人分离失败（上层翻译成 HTTP 500）。"""


@dataclass
class DiarTurn:
    """Pyannote 输出的一个说话人时间片。"""

    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------
_murmurai_mod: Any = None

# 对齐模型按语言缓存。whisperx 的 load_align_model 会按语言下载 wav2vec2（约 1.2GB），
# 原来每个切片都加载一次，长视频等于把同一份权重反复搬进显存。
_ALIGN_CACHE: dict[str, tuple[Any, Any]] = {}
# 本任务内已确认拉取失败的语言，避免每个切片都再试一次
_ALIGN_UNAVAILABLE: set[str] = set()
# 本进程内说话人分离不可用的原因（多为 gated repo 403）。
# 置为非空后不再重复下载 pyannote 权重，修复授权后重启服务即可恢复。
_DIAR_UNAVAILABLE: str = ""


def _clear_align_cache() -> None:
    """任务收尾时释放对齐模型（它们是按语言常驻的，不随单切片销毁）。"""
    _ALIGN_CACHE.clear()
    _ALIGN_UNAVAILABLE.clear()


def _mm():
    """惰性导入 murmurai：模块级 import 会把 ctranslate2/pyannote 一起拉进来，
    拖慢 FastAPI 冷启动（数秒）。"""
    global _murmurai_mod
    if _murmurai_mod is None:
        started = time.time()
        import murmurai  # noqa: PLC0415  (延迟导入是刻意为之)

        _murmurai_mod = murmurai
        logger.info("murmurai 导入完成 %.1fs", time.time() - started)
    return _murmurai_mod


def _filter_kwargs(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """按目标函数签名裁剪参数，兼容 whisperx/murmurai 各版本的参数漂移。"""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    dropped = [k for k in kwargs if k not in params]
    if dropped:
        logger.debug("忽略当前版本不支持的参数: %s", dropped)
    return {k: v for k, v in kwargs.items() if k in params}


def _hf_token() -> str:
    """读取 Hugging Face Token。绝不硬编码，也不写默认值。"""
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
        or os.getenv("HF_HUB_TOKEN")
    )
    if not token:
        raise TranscriptionError(
            "缺少 HF_TOKEN 环境变量。说话人分离需要它，并且必须先在 "
            f"https://hf.co/{DIARIZATION_MODEL} 用同一账号接受许可。"
        )
    return token.strip()


def free_cuda_memory() -> None:
    """强制回收显存：Python 的 GC 不会立刻归还 CUDA 缓存池。"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:  # 某些驱动/容器环境下不可用，忽略即可
            pass


def _resolve_language(language: str | None) -> str | None:
    value = (language or LANGUAGE or "auto").strip().lower()
    return None if value in {"auto", "", "none"} else value


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------
def load_asr_model(model_name: str | None = None) -> Any:
    """加载 faster-whisper 模型（经 murmurai 封装）。

    标准写法（whisperx/murmurai 一致）：
        model = murmurai.load_model("large-v3", device="cuda", compute_type="float16")
    说明：模型每次任务重新加载（large-v3 约 30~60s），任务结束即释放显存。
    若你想常驻复用，改成模块级单例即可，但要自行承担显存占用。
    """
    name = model_name or ASR_MODEL
    mm = _mm()
    kwargs: dict[str, Any] = {
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "language": None,          # 语言在 transcribe() 里按片传入
        "vad_method": VAD_METHOD,  # silero: 轻量、无需许可；pyannote: 需 segmentation 许可
        "asr_options": ASR_OPTIONS,
    }
    if MODEL_DIR:
        # 版本差异：新版叫 download_root，老版叫 model_dir
        sig = inspect.signature(mm.load_model).parameters
        cache_key = "download_root" if "download_root" in sig else ("model_dir" if "model_dir" in sig else None)
        if cache_key:
            kwargs[cache_key] = MODEL_DIR

    logger.info("加载 ASR 模型 %s（device=%s, compute_type=%s, vad=%s）", name, DEVICE, COMPUTE_TYPE, VAD_METHOD)
    started = time.time()
    try:
        model = mm.load_model(name, **_filter_kwargs(mm.load_model, kwargs))
    except Exception as exc:
        raise TranscriptionError(f"ASR 模型加载失败（{name}）: {exc}") from exc
    logger.info("ASR 模型加载完成 %.1fs", time.time() - started)
    return model


def build_diarization_pipeline() -> Any:
    """加载 Pyannote 3.1 说话人分离管线。

    鉴权参数兼容策略（按顺序尝试）：
        1. 若 `Pipeline.from_pretrained` 签名里有 `token`   → 用 token（pyannote.audio >= 4.0）
        2. 否则用 `use_auth_token`（pyannote.audio 3.x）
        3. 实际调用若抛 TypeError，则用另一个参数名再试一次
    Token 一律来自环境变量 HF_TOKEN。
    """
    from pyannote.audio import Pipeline  # noqa: PLC0415 延迟导入

    token = _hf_token()
    model_id = DIARIZATION_MODEL

    try:
        params = inspect.signature(Pipeline.from_pretrained).parameters
        auth_key = "token" if "token" in params else "use_auth_token"
    except (TypeError, ValueError):
        auth_key = "token"

    logger.info("加载说话人分离模型 %s（device=%s, 鉴权参数=%s）", model_id, DEVICE, auth_key)
    started = time.time()
    try:
        pipeline = Pipeline.from_pretrained(model_id, **{auth_key: token})
    except TypeError as exc:
        # 回退：换另一个参数名（不同 pyannote 大版本行为不同）
        fallback = "use_auth_token" if auth_key == "token" else "token"
        logger.warning("pyannote 不接受 %s 参数（%s），回退为 %s", auth_key, exc, fallback)
        pipeline = Pipeline.from_pretrained(model_id, **{fallback: token})
    except Exception as exc:
        raise TranscriptionError(
            f"Pyannote 模型加载失败（{model_id}）: {exc}。"
            "常见原因：HF_TOKEN 无效、未在该模型页接受许可、或网络无法访问 huggingface。"
        ) from exc

    if pipeline is None:
        raise TranscriptionError(
            f"{model_id} 返回空 pipeline：通常是 token 无权限或未接受模型许可。"
        )

    try:
        pipeline.to(torch.device(DEVICE))
    except Exception as exc:  # 极端情况下 pyannote 版本不支持 .to()，退化为 CPU 推理
        logger.warning("pyannote 迁移到 %s 失败（%s），改为 CPU 推理", DEVICE, exc)

    logger.info("说话人分离模型加载完成 %.1fs", time.time() - started)
    return pipeline


def diarize(pipeline: Any, audio_path: str | Path) -> list[DiarTurn]:
    """对整段音频做说话人分离，返回按时间排序的说话人时间片。"""
    kwargs: dict[str, Any] = {}
    if DIAR_MIN_SPEAKERS:
        kwargs["min_speakers"] = int(DIAR_MIN_SPEAKERS)
    if DIAR_MAX_SPEAKERS:
        kwargs["max_speakers"] = int(DIAR_MAX_SPEAKERS)

    audio_path = Path(audio_path)
    logger.info("开始说话人分离: %s（%.1fs 音频）", audio_path.name, probe_duration(audio_path))
    started = time.time()
    try:
        output = pipeline(str(audio_path), **kwargs)
    except Exception as exc:
        raise TranscriptionError(f"说话人分离推理失败: {exc}") from exc

    # pyannote.audio 4.x: DiarizeOutput(speaker_diarization=Annotation, ...)
    annotation = getattr(output, "speaker_diarization", output)
    if not hasattr(annotation, "itertracks"):
        raise TranscriptionError(f"无法解析 Pyannote 输出类型: {type(output)!r}")

    turns = [
        DiarTurn(start=float(turn.start), end=float(turn.end), speaker=str(speaker))
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: (t.start, t.end))
    speakers = {t.speaker for t in turns}
    logger.info(
        "说话人分离完成 %.1fs：%d 段，%d 个说话人 %s",
        time.time() - started, len(turns), len(speakers), sorted(speakers),
    )
    return turns


# ---------------------------------------------------------------------------
# 转录
# ---------------------------------------------------------------------------
def _transcribe_chunk(
    model: Any,
    audio_path: Path,
    batch_size: int,
    language: str | None,
) -> dict:
    """转录单个切片；显存不足时自动减半 batch_size 重试。"""
    attempts = 0
    current_batch = batch_size
    while True:
        try:
            with torch.inference_mode():
                result = model.transcribe(
                    str(audio_path),
                    **_filter_kwargs(
                        model.transcribe,
                        {
                            "batch_size": current_batch,
                            "language": language,
                            "chunk_size": 30,       # whisperx >=3.3 的 VAD 分块时长
                            "print_progress": False,
                        },
                    ),
                )
            return result
        except torch.cuda.OutOfMemoryError as exc:
            if attempts >= 2 or current_batch <= 1:
                raise TranscriptionError(f"显存不足且已降到最小 batch_size: {exc}") from exc
            attempts += 1
            current_batch = max(1, current_batch // 2)
            free_cuda_memory()
            logger.warning("显存不足，batch_size 降为 %d 后重试（第 %d 次）", current_batch, attempts)


def _align_segments(segments: list[dict], audio_path: Path, language: str) -> list[dict]:
    """词级对齐（仅用于「按说话人拆段」）。失败则降级为 segment 级时间戳，不中断任务。

    对齐模型按语言缓存复用：同一任务里 10 个韩语切片只加载一次，而不是每片重来。
    """
    if language in _ALIGN_UNAVAILABLE:
        return segments

    cached = _ALIGN_CACHE.get(language)
    if cached is None:
        mm = _mm()
        try:
            logger.info("加载词级对齐模型（language=%s，首次会自动下载约 1.2GB）", language)
            started = time.time()
            cached = mm.load_align_model(language_code=language, device=DEVICE)
            _ALIGN_CACHE[language] = cached
            logger.info("对齐模型加载完成 %.1fs（language=%s）", time.time() - started, language)
        except Exception as exc:
            _ALIGN_UNAVAILABLE.add(language)
            logger.warning(
                "词级对齐模型不可用（language=%s），本任务跳过对齐、改用 segment 级时间戳: %s",
                language, exc,
            )
            return segments

    align_model, metadata = cached
    try:
        aligned = mm.align(segments, align_model, metadata, str(audio_path), device=DEVICE)
        # whisperx 3.1 返回 list，3.3+ 返回 dict
        return aligned["segments"] if isinstance(aligned, dict) else aligned
    except Exception as exc:
        logger.warning("词级对齐失败（language=%s），降级为 segment 级时间戳: %s", language, exc)
        return segments


def _offset_segments(segments: list[dict], offset: float) -> None:
    """把局部时间戳（相对切片）加上切片起始偏移，还原为全局时间戳。"""
    if offset <= 0:
        return
    for seg in segments:
        seg["start"] = round(float(seg.get("start", 0.0)) + offset, 3)
        seg["end"] = round(float(seg.get("end", seg["start"])) + offset, 3)
        for word in seg.get("words") or []:
            if word.get("start") is not None:
                word["start"] = round(float(word["start"]) + offset, 3)
            if word.get("end") is not None:
                word["end"] = round(float(word["end"]) + offset, 3)


# ---------------------------------------------------------------------------
# 说话人指派 / 切分 / 合并去重
# ---------------------------------------------------------------------------
def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return min(a_end, b_end) - max(a_start, b_start)


def _nearest_speaker(start: float, end: float, turns: Sequence[DiarTurn]) -> str | None:
    """没重叠时（VAD 把静音裁掉了、或切片边界）退化为「时间上最近的说话人」。"""
    best: str | None = None
    best_gap = float("inf")
    for turn in turns:
        if _overlap(start, end, turn.start, turn.end) > 0:
            return turn.speaker
        gap = max(0.0, turn.start - end, start - turn.end)
        if gap < best_gap:
            best, best_gap = turn.speaker, gap
    return best


def _speaker_for_span(start: float, end: float, turns: Sequence[DiarTurn]) -> str | None:
    """取重叠时长最长的说话人。"""
    scores: dict[str, float] = {}
    for turn in turns:
        ov = _overlap(start, end, turn.start, turn.end)
        if ov > 0:
            scores[turn.speaker] = scores.get(turn.speaker, 0.0) + ov
    if not scores:
        return _nearest_speaker(start, end, turns)
    return max(scores.items(), key=lambda kv: kv[1])[0]


def _split_segment_by_words(seg: dict, turns: Sequence[DiarTurn]) -> list[dict] | None:
    """用词级时间戳把一个 segment 按说话人拆成多段（对话场景下很关键）。

    单个词的误判很常见，所以做一次「孤词平滑」：若某个分组只有一个词且
    前后分组是同一说话人，则把它并回前后。
    """
    words = [w for w in (seg.get("words") or []) if w.get("start") is not None and w.get("end") is not None]
    if len(words) < 2:
        return None

    labels: list[str | None] = [
        _speaker_for_span(float(w["start"]), float(w["end"]), turns) for w in words
    ]
    if len({label for label in labels if label}) <= 1:
        return None  # 全段同一人，交给 segment 级处理即可

    # 孤词平滑
    for i in range(1, len(labels) - 1):
        if labels[i] != labels[i - 1] and labels[i - 1] == labels[i + 1]:
            labels[i] = labels[i - 1]

    groups: list[dict] = []
    for word, label in zip(words, labels):
        if groups and groups[-1]["speaker"] == label:
            groups[-1]["words"].append(word)
            groups[-1]["end"] = float(word["end"])
        else:
            groups.append({"speaker": label, "start": float(word["start"]), "end": float(word["end"]), "words": [word]})

    result: list[dict] = []
    for group in groups:
        text = _join_words(group["words"])
        if not text:
            continue
        result.append(
            {
                "start": round(group["start"], 3),
                "end": round(group["end"], 3),
                "text": text,
                "speaker": group["speaker"],
            }
        )
    return result or None


def assign_speakers(segments: list[dict], turns: Sequence[DiarTurn]) -> list[dict]:
    """把 Pyannote 的说话人时间片对齐到 Whisper 的 segment 上。"""
    if not turns:
        for seg in segments:
            seg.setdefault("speaker", "UNKNOWN")
        return segments

    assigned: list[dict] = []
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        # 有词级时间戳就按说话人拆段，否则整段归给重叠最多的说话人
        parts = _split_segment_by_words(seg, turns) if seg.get("words") else None
        if parts:
            assigned.extend(parts)
            continue
        assigned.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "text": str(seg.get("text", "")).strip(),
                "speaker": _speaker_for_span(start, end, turns) or "UNKNOWN",
            }
        )
    return assigned


def _norm_index(text: str) -> tuple[str, list[int]]:
    """归一化（小写、去空白/标点）并保留每个有效字符在原文中的下标。

    保留下标是为了让「重叠字符数」能准确换算成原文的截断位置——
    直接按字符数切原文会因为去掉了空格而错位。
    """
    chars: list[str] = []
    index: list[int] = []
    for i, ch in enumerate(text):
        if ch.isalnum():
            chars.append(ch.lower())
            index.append(i)
    return "".join(chars), index


def _join_text(left: str, right: str) -> str:
    """拼接两段文本，消除切片切点造成的重复（如 "... the pro" + "project ..."）。"""
    left, right = left.strip(), right.strip()
    if not left:
        return right
    if not right:
        return left

    nl, _ = _norm_index(left)
    nr, right_index = _norm_index(right)
    if not nl or not nr:
        return f"{left} {right}"
    if nl == nr:
        return left if len(left) >= len(right) else right
    if nl.endswith(nr) or nr.startswith(nl):
        return left if len(left) >= len(right) else right

    limit = min(len(nl), len(nr), 30)
    for k in range(limit, 2, -1):  # k >= 3 才算重合，避免把常见短词吃掉
        if nl[-k:] == nr[:k]:
            cut = right_index[k - 1] + 1  # 右段前 k 个有效字符在原文中的结束位置
            return (left + right[cut:]).strip()

    # 中日韩字符之间不加空格，拉丁文本之间补空格
    return left + right if (_is_cjk(left[-1]) or _is_cjk(right[0])) else f"{left} {right}"


def _join_words(words: Sequence[dict]) -> str:
    """把 whisperx 的 word 列表拼成文本：首尾空格由这里统一处理，CJK 不加空格。"""
    out = ""
    for item in words:
        token = str(item.get("word", ""))
        if not token:
            continue
        if not out:
            out = token.lstrip()
        elif token[0].isspace() or _is_cjk(out[-1]) or _is_cjk(token.lstrip()[0] if token.lstrip() else " "):
            out += token
        else:
            out += f" {token}"
    return out.strip()


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF      # 汉字
        or 0x3040 <= code <= 0x30FF   # 日文假名
        or 0xAC00 <= code <= 0xD7AF   # 韩文
        or 0x3000 <= code <= 0x303F   # CJK 标点
    )


def postprocess_segments(
    segments: list[dict],
    gap_sec: float = MERGE_GAP_SEC,
    max_merged_sec: float = MAX_MERGED_SEC,
) -> list[dict]:
    """边界合并去重（切片不重叠方案的核心补偿逻辑）。

    规则：
        1. 丢弃空文本；
        2. 相邻两段若「同一说话人」且间隔 <= gap_sec，合并为一段（不超过 max_merged_sec）；
        3. 合并时若文本存在切点重复（重叠前后缀），只保留一份；
        4. 说话人不同则一律不合并 —— 宁可保留碎片，也不能把两个人的话揉成一段。
    """
    cleaned = [
        {
            "start": round(float(s.get("start", 0.0)), 3),
            "end": round(float(s.get("end", 0.0)), 3),
            "speaker": str(s.get("speaker") or "UNKNOWN"),
            "text": str(s.get("text", "")).strip(),
        }
        for s in segments
    ]
    cleaned = [s for s in cleaned if s["text"] and s["end"] >= s["start"]]
    cleaned.sort(key=lambda s: (s["start"], s["end"]))

    merged: list[dict] = []
    for seg in cleaned:
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        gap = seg["start"] - prev["end"]
        same_speaker = seg["speaker"] == prev["speaker"]
        can_grow = (seg["end"] - prev["start"]) <= max_merged_sec
        if same_speaker and gap <= gap_sec and can_grow:
            prev["text"] = _join_text(prev["text"], seg["text"])
            prev["end"] = max(prev["end"], seg["end"])
            continue
        merged.append(seg)

    logger.debug("后处理：%d 段 -> %d 段", len(cleaned), len(merged))
    return merged


def renumber_speakers(segments: list[dict]) -> dict[str, str]:
    """把 SPEAKER_00 之类的原始标签按首次出现顺序重编号为 Speaker 1..N，便于阅读。"""
    mapping: dict[str, str] = {}
    for seg in segments:
        raw = seg["speaker"]
        if raw in {"UNKNOWN", ""}:
            continue
        if raw not in mapping:
            mapping[raw] = f"Speaker {len(mapping) + 1}"
        seg["speaker"] = mapping[raw]
    return mapping


# ---------------------------------------------------------------------------
# 任务级入口
# ---------------------------------------------------------------------------
def transcribe_task(
    chunks: list[AudioChunk],
    language: str | None = None,
    do_diarize: bool = True,
    full_wav: str | Path | None = None,
    progress: Callable[[str, float, str], None] | None = None,
) -> dict:
    """切片列表 -> 带全局时间戳与说话人的 segment 列表。

    :param chunks:     audio.analyze() 的返回值
    :param language:   覆盖 ASR_LANGUAGE，None/'auto' 表示自动判定
    :param do_diarize: 是否做说话人分离（失败会自动降级，不中断转录）
    :param full_wav:   用于全局说话人分离的整段 wav，默认取 chunks[0].source
    :param progress:   进度回调 (stage, fraction 0~1, detail)，用于 /status 实时反馈
    :return: {"language", "segments": [{start,end,speaker,text}], "speakers", ...}
    """
    if not chunks:
        raise TranscriptionError("切片列表为空，无法转录")

    def report(stage: str, fraction: float, detail: str = "") -> None:
        """进度上报失败绝不影响主流程。"""
        if progress is None:
            return
        try:
            progress(stage, max(0.0, min(1.0, fraction)), detail)
        except Exception:  # noqa: BLE001
            logger.debug("进度回调异常", exc_info=True)

    lang = _resolve_language(language)
    source_wav = Path(full_wav) if full_wav else (chunks[0].source or chunks[0].path)
    total_duration = round(sum(c.duration_sec for c in chunks), 3)

    model: Any = None
    diar_pipeline: Any = None
    turns: list[DiarTurn] = []
    detected_languages: list[str] = []
    collected: list[dict] = []
    lang_votes: Counter[str] = Counter()
    fixed_lang: str | None = lang
    started = time.time()

    global _DIAR_UNAVAILABLE

    try:
        model = load_asr_model()

        if do_diarize and _DIAR_UNAVAILABLE:
            # 已经确认拉不起 pyannote（多为 gated repo 403）：本进程内不再重复下载权重
            logger.info(
                "说话人分离在本进程内已标记为不可用，直接跳过（原因：%s）。修复授权后请重启服务",
                _DIAR_UNAVAILABLE,
            )
            report("diarizing", 1.0, "说话人分离不可用（本次已跳过）")
        elif do_diarize:
            try:
                report("diarizing", 0.0, "加载说话人分离模型")
                diar_pipeline = build_diarization_pipeline()
                turns = diarize(diar_pipeline, source_wav)
                report("diarizing", 1.0, f"识别到 {len({t.speaker for t in turns})} 位说话人")
            except Exception as exc:
                _DIAR_UNAVAILABLE = str(exc)[:500]
                logger.error("说话人分离失败，本次结果将不含说话人标签: %s", exc)
                report("diarizing", 1.0, "说话人分离失败，已降级为无标签转录")
                turns = []

        total_chunks = len(chunks)
        for chunk in chunks:
            report("transcribing", chunk.index / total_chunks, f"转录切片 {chunk.index + 1}/{total_chunks}")
            logger.info("转录切片 %d/%d（start=%.1fs, %.1fs）", chunk.index + 1, len(chunks), chunk.start_sec, chunk.duration_sec)
            t0 = time.time()
            result = _transcribe_chunk(model, chunk.path, BATCH_SIZE, fixed_lang)

            chunk_language = result.get("language") or fixed_lang or "en"
            detected_languages.append(chunk_language)
            segments = [dict(s) for s in (result.get("segments") or [])]

            # 前 N 个「有语音」的切片投票决定全局语言，之后固定传参：
            # 逐片 auto 会让同一文件的语言来回跳（前 20 分钟 en、后面 ko），而且每片多花一次检测。
            if fixed_lang is None and segments:
                lang_votes[chunk_language] += 1
                if sum(lang_votes.values()) >= LANG_VOTE_SAMPLES:
                    fixed_lang = lang_votes.most_common(1)[0][0]
                    logger.info(
                        "语言投票确定：%s（票数 %s），后续切片固定使用", fixed_lang, dict(lang_votes)
                    )

            # 对齐结果只用于「按说话人拆段」：分离失败时 words 根本不会被读取，
            # 却会触发 1.2GB/语言的 wav2vec2 下载 —— auto 模式下直接跳过。
            if ALIGN_ENABLED and segments and (turns or not ALIGN_ONLY_WITH_DIARIZATION):
                segments = _align_segments(segments, chunk.path, chunk_language)

            _offset_segments(segments, chunk.start_sec)

            local_turns = [
                t for t in turns
                if t.end >= chunk.start_sec - 1.0 and t.start <= chunk.end_sec + 1.0
            ]
            segments = assign_speakers(segments, local_turns)
            for seg in segments:
                seg["chunk_index"] = chunk.index
            collected.extend(segments)
            logger.info("切片 %d 完成：%d 段，耗时 %.1fs", chunk.index, len(segments), time.time() - t0)
            free_cuda_memory()

        if not collected:
            raise TranscriptionError("所有切片都没有转录出文本，请检查音频是否含有效语音")

        report("transcribing", 1.0, "转录与对齐完成，开始后处理")
        diarized = bool(turns)
        gap_sec = MERGE_GAP_SEC if diarized else MERGE_GAP_SEC_NO_DIAR
        if not diarized:
            logger.info(
                "未获得说话人标签：合并窗口收紧为 %.1fs（默认 %.1fs），避免把不同人的话揉成一段",
                gap_sec, MERGE_GAP_SEC,
            )
        segments = postprocess_segments(collected, gap_sec=gap_sec)
        mapping = renumber_speakers(segments)

        result_payload = {
            "language": Counter(detected_languages).most_common(1)[0][0] if detected_languages else (lang or "auto"),
            "segments": segments,
            "speakers": [mapping[k] for k in sorted(mapping, key=lambda k: int(mapping[k].split()[-1]))],
            "num_segments": len(segments),
            "duration_sec": total_duration,
            "diarized": bool(turns),
            "elapsed_sec": round(time.time() - started, 1),
        }
        logger.info(
            "转录完成：%d 段 / %d 位说话人 / 总耗时 %.1fs",
            result_payload["num_segments"], len(result_payload["speakers"]), result_payload["elapsed_sec"],
        )
        return result_payload

    finally:
        # 显存释放：显式 del + empty_cache（murmurai / pyannote 都不保证自己清理）
        if model is not None:
            del model
        if diar_pipeline is not None:
            try:
                diar_pipeline.to(torch.device("cpu"))
            except Exception:
                pass
            del diar_pipeline
        _clear_align_cache()
        free_cuda_memory()
        logger.info("模型与显存已释放（可用显存 %.0f MB）", _available_vram_mb())


def _available_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        free, _ = torch.cuda.mem_get_info()
        return free / 1024 / 1024
    except Exception:
        return 0.0


def transcribe_file(
    media_path: str | Path,
    chunk_minutes: float | None = None,
    language: str | None = None,
    do_diarize: bool = True,
    cleanup_audio: bool = False,
) -> dict:
    """端到端便利函数：媒体文件 -> 转录结果（内部先调 audio.analyze）。"""
    chunks = analyze(media_path, chunk_minutes=chunk_minutes)
    try:
        return transcribe_task(chunks, language=language, do_diarize=do_diarize)
    finally:
        if cleanup_audio:
            cleanup(chunks, remove_source=True)


# ---------------------------------------------------------------------------
# CLI：python -m app.services.transcribe <file> [--no-diarize] [--out x.json]
# ---------------------------------------------------------------------------
def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="murmurai 转录 + Pyannote 说话人分离")
    parser.add_argument("input", help="视频/音频路径")
    parser.add_argument("--chunk-minutes", type=float, default=None)
    parser.add_argument("--language", default=None, help="auto / en / ja / ko ...")
    parser.add_argument("--no-diarize", action="store_true", help="跳过说话人分离（无需 HF_TOKEN）")
    parser.add_argument("--out", default=None, help="结果 JSON 输出路径")
    parser.add_argument("--print", type=int, default=10, help="打印前 N 段")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    payload = transcribe_file(
        args.input,
        chunk_minutes=args.chunk_minutes,
        language=args.language,
        do_diarize=not args.no_diarize,
    )
    for seg in payload["segments"][: args.print]:
        print(f"[{seg['start']:>8.2f} - {seg['end']:>8.2f}] {seg['speaker']:<10} {seg['text']}")

    if args.out:
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {args.out}")
    else:
        print(json.dumps({k: v for k, v in payload.items() if k != "segments"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
