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
    - pyannote.audio 4.x 默认用 torchcodec 解码音频，而 torchcodec 必须与 torch
      版本严格匹配、且 Windows 上还需要共享版 FFmpeg DLL。因此默认**不走路径**，
      而是自己把 16k PCM wav 读成 `{"waveform", "sample_rate"}` 再传给 pipeline
      （`DIAR_AUDIO_INPUT=path` 可退回旧行为）。
"""

from __future__ import annotations

import gc
import inspect
import json
import logging
import os
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from app.services.audio import AudioChunk, analyze, cleanup, probe_duration

# torch 2.9 起旧的 TF32 开关被弃用，pyannote / lightning 仍在读它，每次启动都会刷一条
# UserWarning。只按 message 精确屏蔽这一条（torchcodec / HF 权限等警告必须保留）。
# 注意不能加 module= 限制：该警告的归属模块是调用方（pyannote），不是 torch.backends.cuda。
warnings.filterwarnings(
    "ignore",
    message=r".*TF32 behavior.*",
    category=UserWarning,
)

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

# 模型加载是否需要联网：
#   auto（默认）—— 先常规加载；遇到网络类错误时切到 HF 离线模式重试一次，
#                  模型权重已缓存即可直接恢复（典型场景：HF 504 / 代理挂了）
#   1            —— 始终离线（内网/容器，模型已预热）
#   0            —— 始终联网，不做离线重试
ASR_OFFLINE = (os.getenv("ASR_OFFLINE", "auto") or "auto").strip().lower()

# 词级对齐开关：
#   1 / true（默认）—— 需要说话人标签时才对齐（对齐结果只用于「按说话人拆段」）。
#   0 / false       —— 完全关闭。
# 注意：没有说话人标签时一律跳过。words 不会被写入 transcript.json，此时对齐
# 没有任何消费方，却会触发 1.2GB/语言的 wav2vec2 下载（分离失败后尤其浪费）。
ALIGN_MODE = (os.getenv("ASR_ALIGN", "1") or "1").strip().lower()
ALIGN_ENABLED = ALIGN_MODE not in {"0", "false", "no", "off"}

DIARIZATION_MODEL = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
DIAR_MIN_SPEAKERS = os.getenv("DIAR_MIN_SPEAKERS")  # 可选：已知说话人数时提升准确率
DIAR_MAX_SPEAKERS = os.getenv("DIAR_MAX_SPEAKERS")

# 说话人分离的音频输入形态：
#   waveform（默认）—— 自己解码 16k PCM wav 并传 {"waveform", "sample_rate"}，
#                     完全不依赖 torchcodec（4.x 上最常见的失败源）
#   path             —— 直接把路径交给 pyannote，需要 torchcodec 可用
DIAR_AUDIO_INPUT = (os.getenv("DIAR_AUDIO_INPUT", "waveform") or "waveform").strip().lower()
# 整段 waveform 会常驻内存/显存：约 230MB/小时（16k float32）。超过该时长只告警。
DIAR_MAX_HOURS = float(os.getenv("DIAR_MAX_HOURS", "2"))
# 分离前裁掉首尾静音（长静音开头/结尾很常见，pyannote 对它们做滑窗纯属浪费）
DIAR_TRIM_SILENCE = (os.getenv("DIAR_TRIM_SILENCE", "1") or "1").strip().lower() not in {
    "0", "false", "no", "off",
}
DIAR_TRIM_MIN_SEC = float(os.getenv("DIAR_TRIM_MIN_SEC", "1.0"))  # 裁剪后至少保留的语音时长
DIAR_ENERGY_THRESHOLD = float(os.getenv("DIAR_ENERGY_THRESHOLD", "0.001"))  # ≈ -60dBFS

# 分离结果清洗：总时长过短的「幽灵说话人」（BGM/音效误聚类）并入最近的保留说话人
DIAR_MIN_SPEAKER_SEC = float(os.getenv("DIAR_MIN_SPEAKER_SEC", "5"))
DIAR_MIN_SPEAKER_SHARE = float(os.getenv("DIAR_MIN_SPEAKER_SHARE", "0.01"))
# 永久性错误（403 / token）才进程级缓存；临时错误（OOM/网络）下一个任务重试
DIAR_CACHE_FAILURE = (os.getenv("DIAR_CACHE_FAILURE", "1") or "1").strip().lower() not in {
    "0", "false", "no", "off",
}
_PERMANENT_DIAR_SIGNS = (
    "gated repo", "not in the authorized list", "401", "403", "invalid token",
    "unauthorized", "token is required", "torchcodec is not available",
)

# 说话人指派：segment 与最近的 diarization turn 相距超过该值就标 UNKNOWN，
# 否则音乐/空白区会被贴上几十秒外的说话人（宁可没标签，也不要贴错人）。
MAX_SPEAKER_ATTACH_GAP = float(os.getenv("MAX_SPEAKER_ATTACH_GAP", "2.0"))

MERGE_GAP_SEC = float(os.getenv("MERGE_GAP_SEC", "0.8"))       # 同说话人合并阈值
# 没有说话人标签时，所有人都是 UNKNOWN，若沿用 0.8s 会把不同人的话揉进同一段
MERGE_GAP_SEC_NO_DIAR = float(os.getenv("MERGE_GAP_SEC_NO_DIAR", "0.3"))
MAX_MERGED_SEC = float(os.getenv("MAX_MERGED_SEC", "60"))        # 合并后单段上限，防止整集变一段

# Whisper 幻觉过滤（BGM / 静音处的高频「感谢观看」类输出）
HALLU_NO_SPEECH_PROB = float(os.getenv("HALLU_NO_SPEECH_PROB", "0.6"))
HALLU_AVG_LOGPROB = float(os.getenv("HALLU_AVG_LOGPROB", "-1.0"))
HALLU_COMPRESSION_RATIO = float(os.getenv("HALLU_COMPRESSION_RATIO", "2.4"))

# 前 N 个「有语音」的切片各投一票决定全局语言，之后固定传给所有切片。
# 逐片 auto 检测会让同一文件的语言来回跳（例如前 20 分钟被判成 en、之后是 ko）。
LANG_VOTE_SAMPLES = max(1, int(os.getenv("ASR_LANG_VOTE_SAMPLES", "3")))
# 只有当某个语言占比达到该阈值时才锁定；多语内容/平票继续逐片检测。
# 误锁的代价远大于每片多一次语言检测（约 1~2s），所以默认取 0.8。
LANG_VOTE_MIN_SHARE = float(os.getenv("ASR_LANG_MIN_SHARE", "0.8"))

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
# 本进程内说话人分离不可用的原因（只缓存永久性错误，多为 gated repo 403）。
# 置为非空后不再重复下载 pyannote 权重，修复授权后重启服务即可恢复。
_DIAR_UNAVAILABLE: str = ""
# 最近一次分离失败的原因（含临时错误），用于 /healthz 与 transcript.json 的可诊断性
_DIAR_LAST_ERROR: str = ""


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


def _is_permanent_diar_error(text: str) -> bool:
    """区分「配置/授权错误」与「临时错误（OOM、网络抖动）」。"""
    lowered = text.lower()
    return any(sign in lowered for sign in _PERMANENT_DIAR_SIGNS)


def diarization_status() -> dict[str, Any]:
    """说话人分离可用性快照（不含密钥），供 /healthz 与 /status 使用。"""
    return {
        "input_mode": DIAR_AUDIO_INPUT,
        "available": not _DIAR_UNAVAILABLE,
        "cached_reason": _DIAR_UNAVAILABLE or None,
        "last_error": _DIAR_LAST_ERROR or None,
    }


def reset_diarization_cache() -> None:
    """清空「分离不可用」缓存（例如装好 torchcodec / 重新 Agree 之后）。"""
    global _DIAR_UNAVAILABLE, _DIAR_LAST_ERROR
    _DIAR_UNAVAILABLE, _DIAR_LAST_ERROR = "", ""


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
# 离线兜底（HuggingFace 504 / 代理不可用时，已缓存的模型仍可加载）
# ---------------------------------------------------------------------------
_NETWORK_ERROR_SIGNS = (
    "connection", "connect", "timed out", "timeout", "time-out", "gateway",
    "502", "503", "504", "ssl", "proxy", "getaddrinfo", "name or service not known",
    "temporary failure", "network is unreachable", "max retries exceeded",
)


def _looks_like_network_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(sign in text for sign in _NETWORK_ERROR_SIGNS)


def _enable_offline_mode() -> None:
    """强制 huggingface_hub 只读本地缓存（不再做远端元数据校验）。

    注意：huggingface_hub 在 import 时就把 HF_HUB_OFFLINE 读进常量，
    因此除了设环境变量，还要把常量一起改掉，运行中切换才立即生效。
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import huggingface_hub.constants as hf_constants  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return
    for attr in ("HF_HUB_OFFLINE", "HF_HUB_OFFLINE_MODE"):
        if hasattr(hf_constants, attr):
            setattr(hf_constants, attr, True)


def _model_load_hint(name: str, exc: Exception) -> str:
    """把模型加载错误翻译成可操作的提示（区分网络问题与其它问题）。"""
    if _looks_like_network_error(exc):
        return (
            f"ASR 模型加载失败（{name}）: {exc}。"
            "这是下载/校验模型时的网络错误（HuggingFace 或代理不可达），不是代码问题。"
            "模型已缓存时：在 .env 设 ASR_OFFLINE=1（或 HF_HUB_OFFLINE=1）后重启即可离线加载；"
            "需要重新下载时：检查代理/VPN，或设 HF_ENDPOINT=https://hf-mirror.com。"
        )
    return f"ASR 模型加载失败（{name}）: {exc}"


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

    if ASR_OFFLINE == "1":
        _enable_offline_mode()

    logger.info("加载 ASR 模型 %s（device=%s, compute_type=%s, vad=%s）", name, DEVICE, COMPUTE_TYPE, VAD_METHOD)
    started = time.time()
    call_kwargs = _filter_kwargs(mm.load_model, kwargs)
    try:
        model = mm.load_model(name, **call_kwargs)
    except Exception as exc:
        # 网络类错误且处于 auto 模式：切离线重试一次（权重已缓存时能直接恢复）
        if ASR_OFFLINE != "auto" or not _looks_like_network_error(exc):
            raise TranscriptionError(_model_load_hint(name, exc)) from exc

        logger.warning("ASR 模型加载遇到网络错误（%s），切换 HF 离线模式重试一次", exc)
        _enable_offline_mode()
        logger.warning("本进程后续模型加载将只读本地缓存（重启服务后恢复联网）")
        try:
            model = mm.load_model(name, **call_kwargs)
        except Exception as retry_exc:
            raise TranscriptionError(_model_load_hint(name, retry_exc)) from retry_exc
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


def _load_waveform(path: Path, target_sr: int = 16000) -> tuple[torch.Tensor, int]:
    """把 16k wav 读成 pyannote 4.x 需要的 waveform 输入。

    为什么不用路径：pyannote.audio 4.x 用 torchcodec 解码音频，而 torchcodec
    必须与 torch 版本严格匹配、Windows 上还需要共享版 FFmpeg DLL，本地几乎必然
    报「torchcodec is not available」。本项目已经用 FFmpeg 统一转成 16k PCM，
    自己读一遍最稳，且不新增任何依赖（soundfile 随 pyannote/murmurai 一起装）。
    """
    path = Path(path)
    try:
        import soundfile as sf  # noqa: PLC0415 延迟导入
    except ImportError:
        return _load_waveform_stdlib(path, target_sr)

    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:  # 损坏/非 WAV 时退回标准库再试一次
        logger.warning("soundfile 读取失败（%s），改用标准库 wave 兜底", exc)
        return _load_waveform_stdlib(path, target_sr)

    if data.size == 0:
        raise TranscriptionError(f"音频为空，无法做说话人分离: {path}")
    if sr != target_sr:
        raise TranscriptionError(f"采样率不是 {target_sr}Hz（实际 {sr}）: {path}")

    waveform = torch.from_numpy(data.T).contiguous()  # (channel, time)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sr


def _load_waveform_stdlib(path: Path, target_sr: int) -> tuple[torch.Tensor, int]:
    """soundfile 缺失时的兜底：本项目生成的 wav 一定是 16bit PCM。"""
    import wave  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        sr = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    if sr != target_sr:
        raise TranscriptionError(f"采样率不是 {target_sr}Hz（实际 {sr}）: {path}")
    if width != 2:
        raise TranscriptionError(f"仅支持 16bit PCM wav（实际 {width * 8}bit）: {path}")

    audio = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return torch.from_numpy(audio).unsqueeze(0), sr


def _speech_span(waveform: torch.Tensor, sr: int) -> tuple[float, float] | None:
    """返回首个/最后一个「有能量」采样点覆盖的时间段（前后各留 1s 余量）。

    用于在分离前裁掉首尾静音：84 分钟素材里可能有 20 分钟静音，
    对它们做 pyannote 滑窗纯属浪费（实测约占总耗时的 25%）。
    """
    energy = waveform.abs().amax(dim=0)
    index = torch.nonzero(energy > DIAR_ENERGY_THRESHOLD).flatten()
    if index.numel() == 0:
        return None

    margin = sr  # 前后各留 1s，避免切掉实际的起音/尾音
    start = max(0, int(index[0]) - margin)
    end = min(waveform.size(1), int(index[-1]) + 1 + margin)
    if (end - start) / sr < DIAR_TRIM_MIN_SEC:
        return None
    return start / sr, end / sr


def _diarization_input(audio_path: Path) -> tuple[Any, float]:
    """生成 pipeline 输入，返回 (payload, 时间偏移秒)。

    payload 为 None 表示整段没有有效语音（调用方直接返回空 turns）。
    waveform 模式会裁掉首尾静音，因此返回的 turns 需要加上这个偏移。
    """
    if DIAR_AUDIO_INPUT == "path":
        return str(audio_path), 0.0

    try:
        waveform, sr = _load_waveform(audio_path)
    except TranscriptionError:
        raise
    except Exception as exc:
        raise TranscriptionError(f"读取音频失败（{audio_path.name}）: {exc}") from exc

    total_sec = waveform.size(1) / sr
    offset = 0.0
    if DIAR_TRIM_SILENCE:
        span = _speech_span(waveform, sr)
        if span is None:
            return None, 0.0
        start, end = span
        if start > 0 or end < total_sec:
            waveform = waveform[:, int(start * sr) : int(end * sr)]
            offset = start
            logger.info(
                "分离前裁掉首尾静音：偏移 %.1fs，保留 %.1fs / %.1fs（%.0f MB）",
                start, end - start, total_sec, waveform.numel() * 4 / 1e6,
            )

    logger.info(
        "说话人分离输入：waveform（%.0f MB, sr=%d）", waveform.numel() * 4 / 1e6, sr
    )
    return {"waveform": waveform, "sample_rate": sr}, offset


def _opt_int(raw: str | None) -> int | None:
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


def diarize(
    pipeline: Any,
    audio_path: str | Path,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[DiarTurn]:
    """对整段音频做说话人分离，返回按时间排序的说话人时间片。"""
    kwargs: dict[str, Any] = {}
    min_count = min_speakers or _opt_int(DIAR_MIN_SPEAKERS)
    max_count = max_speakers or _opt_int(DIAR_MAX_SPEAKERS)
    if min_count:
        kwargs["min_speakers"] = int(min_count)
    if max_count:
        kwargs["max_speakers"] = int(max_count)

    audio_path = Path(audio_path)
    duration = probe_duration(audio_path)
    logger.info("开始说话人分离: %s（%.1fs 音频）", audio_path.name, duration)
    if duration / 3600 > DIAR_MAX_HOURS:
        logger.warning(
            "音频 %.1fh 超过 DIAR_MAX_HOURS=%.1f：waveform 将常驻约 %.0f MB 内存/显存",
            duration / 3600, DIAR_MAX_HOURS, duration * 16000 * 4 / 1e6,
        )

    payload, offset = _diarization_input(audio_path)
    if payload is None:
        logger.warning("整段音频没有检测到有效语音，跳过说话人分离: %s", audio_path.name)
        return []

    started = time.time()
    try:
        output = pipeline(payload, **kwargs)
    except TranscriptionError:
        raise
    except Exception as exc:
        raise TranscriptionError(f"说话人分离推理失败: {exc}") from exc

    # pyannote.audio 4.x: DiarizeOutput(speaker_diarization=Annotation, ...)
    annotation = getattr(output, "speaker_diarization", output)
    if not hasattr(annotation, "itertracks"):
        raise TranscriptionError(f"无法解析 Pyannote 输出类型: {type(output)!r}")

    turns = [
        DiarTurn(
            start=round(float(turn.start) + offset, 3),
            end=round(float(turn.end) + offset, 3),
            speaker=str(speaker),
        )
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: (t.start, t.end))
    speakers = {t.speaker for t in turns}
    logger.info(
        "说话人分离完成 %.1fs：%d 段，%d 个说话人 %s",
        time.time() - started, len(turns), len(speakers), sorted(speakers),
    )
    return turns


def _gap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """两个时间片之间的间隔（重叠时为 0）。"""
    return max(0.0, b_start - a_end, a_start - b_end)


def _merge_short_speakers(turns: list[DiarTurn], speech_duration: float) -> list[DiarTurn]:
    """把总时长过短的「幽灵说话人」并入时间上最近的保留说话人。

    pyannote 3.1 在 BGM / 音效多的素材上容易额外聚出只说了两三秒的说话人，
    留着会让总结里出现「6 位说话人」这种明显错误的结果。

    :param speech_duration: 有效语音总时长（不是文件总时长）。
        用文件总时长算占比会偏高：84 分钟素材里含 20 分钟静音时，
        1% 阈值会从 39s 被抬到 51s，真实的短发言者会被误合并。
    """
    if len({t.speaker for t in turns}) <= 1:
        return turns

    durations: dict[str, float] = {}
    for turn in turns:
        durations[turn.speaker] = durations.get(turn.speaker, 0.0) + turn.duration

    floor = max(DIAR_MIN_SPEAKER_SEC, speech_duration * DIAR_MIN_SPEAKER_SHARE)
    kept = {name for name, seconds in durations.items() if seconds >= floor}
    if len(kept) == len(durations):
        return turns
    if len(kept) < 2:
        # 过滤后只剩一个人：多半是单人素材或分离失败，保留原始标签更有诊断价值
        logger.warning(
            "说话人清洗会只剩 %d 位（阈值 %.1fs），跳过合并", len(kept), floor
        )
        return turns

    kept_turns = [t for t in turns if t.speaker in kept]
    merged: list[DiarTurn] = []
    remapped: dict[str, str] = {}
    for turn in turns:
        if turn.speaker in kept:
            merged.append(turn)
            continue
        nearest = min(
            kept_turns, key=lambda k: _gap(turn.start, turn.end, k.start, k.end)
        )
        remapped.setdefault(turn.speaker, nearest.speaker)
        merged.append(DiarTurn(start=turn.start, end=turn.end, speaker=nearest.speaker))

    logger.info(
        "说话人清洗：%d -> %d 位（合并 %s）", len(durations), len(kept), remapped
    )
    return merged


def _speaker_stats(
    turns: Sequence[DiarTurn],
    mapping: dict[str, str] | None = None,
) -> list[dict]:
    """每位说话人的总时长与段数，用于判断是否过度聚类（`Speaker 4` 只有 3 分钟时基本是幽灵）。

    :param mapping: renumber_speakers() 的 SPEAKER_xx -> Speaker N 映射，
        传入后统计结果与 transcript.json 里的 speakers 一致。
    """
    stats: dict[str, dict] = {}
    for turn in turns:
        if mapping is not None and turn.speaker not in mapping:
            # 该说话人没有任何被转录到的片段（纯音效/被 VAD 裁掉），不参与统计
            continue
        name = mapping[turn.speaker] if mapping is not None else turn.speaker
        item = stats.setdefault(name, {"speaker": name, "seconds": 0.0, "turns": 0})
        item["seconds"] = round(item["seconds"] + turn.duration, 1)
        item["turns"] += 1
    return sorted(stats.values(), key=lambda item: -item["seconds"])


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

    # 必须在缓存命中分支之外取模块句柄：否则第二个同语言切片会因 mm 未定义而必然失败
    mm = _mm()

    cached = _ALIGN_CACHE.get(language)
    if cached is None:
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
        # 同语言的后续切片不再重复尝试，避免日志刷屏与重复耗时
        _ALIGN_UNAVAILABLE.add(language)
        logger.warning(
            "词级对齐失败（language=%s），本任务剩余切片改用 segment 级时间戳: %s", language, exc
        )
        return segments


def _as_float(value: Any) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _norm_text(text: str) -> str:
    """归一化文本用于重复检测：只保留字母/数字，忽略大小写与标点。"""
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _filter_hallucinations(segments: list[dict], label: str = "") -> list[dict]:
    """过滤 Whisper 在 BGM / 静音处的典型幻觉。

    三类规则（阈值均可用环境变量覆盖）：
        1. compression_ratio 过高 —— 复读机式输出；
        2. no_speech_prob 高且 avg_logprob 低 —— 这段其实没人说话；
        3. 与上一段归一化后完全相同 —— 连续重复字幕。
    少于 6 个有效字符的重复（例如「はい」「OK」）不算幻觉，避免误杀。
    """
    kept: list[dict] = []
    dropped = {"compression": 0, "no_speech": 0, "duplicate": 0}
    previous = ""

    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue

        compression = _as_float(seg.get("compression_ratio"))
        if compression is not None and compression > HALLU_COMPRESSION_RATIO:
            dropped["compression"] += 1
            continue

        no_speech = _as_float(seg.get("no_speech_prob"))
        logprob = _as_float(seg.get("avg_logprob"))
        if (
            no_speech is not None and no_speech >= HALLU_NO_SPEECH_PROB
            and logprob is not None and logprob <= HALLU_AVG_LOGPROB
        ):
            dropped["no_speech"] += 1
            continue

        normalized = _norm_text(text)
        if len(normalized) >= 6 and normalized == previous:
            dropped["duplicate"] += 1
            continue
        previous = normalized

        kept.append(seg)

    total_dropped = sum(dropped.values())
    if total_dropped:
        logger.info(
            "幻觉过滤%s：丢弃 %d 段（%s）",
            f"[{label}]" if label else "",
            total_dropped,
            {k: v for k, v in dropped.items() if v},
        )
    return kept


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
    """没重叠时（VAD 把静音裁掉了、或切片边界）退化为「时间上最近的说话人」。

    但距离必须有上限：音乐/空白区的 segment 若被贴上 60 秒外的人，会被
    postprocess 合并进那个人的段落里。宁可返回 None（最终标 UNKNOWN）。
    """
    best: str | None = None
    best_gap = float("inf")
    for turn in turns:
        if _overlap(start, end, turn.start, turn.end) > 0:
            return turn.speaker
        gap = _gap(start, end, turn.start, turn.end)
        if gap < best_gap:
            best, best_gap = turn.speaker, gap
    return best if best_gap <= MAX_SPEAKER_ATTACH_GAP else None


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


def _fill_missing_labels(labels: list[str | None]) -> list[str | None]:
    """用前后邻近的标签补齐「离所有说话人时间片都太远」的词，避免段内出现 UNKNOWN 碎片。"""
    filled = list(labels)
    last: str | None = None
    for i, label in enumerate(filled):
        if label is None:
            filled[i] = last
        else:
            last = label

    nxt: str | None = None
    for i in range(len(filled) - 1, -1, -1):
        if filled[i] is None:
            filled[i] = nxt
        else:
            nxt = filled[i]
    return filled


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
    if not any(labels):
        return None  # 整段都离所有说话人时间片太远，交给 segment 级标 UNKNOWN
    labels = _fill_missing_labels(labels)
    if len({label for label in labels}) <= 1:
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
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> dict:
    """切片列表 -> 带全局时间戳与说话人的 segment 列表。

    :param chunks:      audio.analyze() 的返回值
    :param language:    覆盖 ASR_LANGUAGE，None/'auto' 表示自动判定
    :param do_diarize:  是否做说话人分离（失败会自动降级，不中断转录）
    :param full_wav:    用于全局说话人分离的整段 wav，默认取 chunks[0].source
    :param progress:    进度回调 (stage, fraction 0~1, detail)，用于 /status 实时反馈
    :param min_speakers: 已知人数时传入，显著提升分离准确率（覆盖 DIAR_MIN/MAX_SPEAKERS）
    :param max_speakers: 同上
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
    diar_speech_sec = 0.0
    detected_languages: list[str] = []
    collected: list[dict] = []
    chunk_stats: list[dict] = []
    lang_votes: Counter[str] = Counter()
    fixed_lang: str | None = lang
    started = time.time()

    global _DIAR_UNAVAILABLE, _DIAR_LAST_ERROR

    if do_diarize:
        _DIAR_LAST_ERROR = ""  # 只保留本任务的失败原因

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
                turns = diarize(
                    diar_pipeline, source_wav,
                    min_speakers=min_speakers, max_speakers=max_speakers,
                )
                # 清洗阈值用「有效语音时长」而不是文件总时长：长静音会把 1% 门槛抬高
                diar_speech_sec = round(sum(t.duration for t in turns), 1)
                turns = _merge_short_speakers(turns, diar_speech_sec)
                if total_duration and diar_speech_sec:
                    logger.info(
                        "说话人覆盖：%.0fs / 总时长 %.0fs（%.1f%%）",
                        diar_speech_sec, total_duration, diar_speech_sec / total_duration * 100,
                    )
                report("diarizing", 1.0, f"识别到 {len({t.speaker for t in turns})} 位说话人")
            except Exception as exc:
                _DIAR_LAST_ERROR = str(exc)[:500]
                permanent = _is_permanent_diar_error(_DIAR_LAST_ERROR)
                if DIAR_CACHE_FAILURE and permanent:
                    _DIAR_UNAVAILABLE = _DIAR_LAST_ERROR
                    logger.error("说话人分离不可用（永久性错误，本进程内不再重试）: %s", exc)
                else:
                    logger.error("说话人分离失败（临时错误，下个任务会重试）: %s", exc)
                report("diarizing", 1.0, "说话人分离失败，已降级为无标签转录")
                turns = []
            finally:
                # pyannote 与 ASR 模型同时常驻会多占约 2GB 显存，分离完立刻释放
                if diar_pipeline is not None:
                    try:
                        diar_pipeline.to(torch.device("cpu"))
                    except Exception:
                        logger.debug("pyannote 迁回 CPU 失败", exc_info=True)
                    del diar_pipeline
                    diar_pipeline = None
                    free_cuda_memory()

        total_chunks = len(chunks)
        for chunk in chunks:
            report("transcribing", chunk.index / total_chunks, f"转录切片 {chunk.index + 1}/{total_chunks}")
            logger.info("转录切片 %d/%d（start=%.1fs, %.1fs）", chunk.index + 1, len(chunks), chunk.start_sec, chunk.duration_sec)
            t0 = time.time()
            result = _transcribe_chunk(model, chunk.path, BATCH_SIZE, fixed_lang)

            chunk_language = result.get("language") or fixed_lang or "unknown"
            detected_languages.append(chunk_language)
            segments = _filter_hallucinations(
                [dict(s) for s in (result.get("segments") or [])],
                label=f"切片 {chunk.index}",
            )

            # 只有「明显单语」时才锁定全局语言；多语内容/平票继续逐片检测。
            # 强行锁成单一语言会把英文/韩文段按错误语言解码，错误会被放大到整片。
            if fixed_lang is None and segments:
                lang_votes[chunk_language] += 1
                total_votes = sum(lang_votes.values())
                top, count = lang_votes.most_common(1)[0]
                if total_votes >= LANG_VOTE_SAMPLES:
                    if count / total_votes >= LANG_VOTE_MIN_SHARE:
                        fixed_lang = top
                        logger.info("语言投票锁定：%s（票数 %s）", fixed_lang, dict(lang_votes))
                    else:
                        logger.info(
                            "多语内容（票数 %s），不锁定语言，继续逐片检测", dict(lang_votes)
                        )

            # 对齐结果只用于「按说话人拆段」：没有说话人标签时 words 根本不会被读取，
            # 却要按语言各下载/加载一次 1.2GB wav2vec2，因此一律跳过。
            align_language = fixed_lang or chunk_language
            if ALIGN_ENABLED and segments and turns and align_language != "unknown":
                segments = _align_segments(segments, chunk.path, align_language)

            _offset_segments(segments, chunk.start_sec)

            local_turns = [
                t for t in turns
                if t.end >= chunk.start_sec - 1.0 and t.start <= chunk.end_sec + 1.0
            ]
            segments = assign_speakers(segments, local_turns)
            for seg in segments:
                seg["chunk_index"] = chunk.index
            collected.extend(segments)
            chunk_stats.append(
                {
                    "index": chunk.index,
                    "start": round(chunk.start_sec, 1),
                    "duration": round(chunk.duration_sec, 1),
                    "language": chunk_language,
                    "segments": len(segments),
                }
            )
            logger.info("切片 %d 完成：%d 段，耗时 %.1fs", chunk.index, len(segments), time.time() - t0)
            if not segments:
                logger.warning(
                    "切片 %d（%.0f~%.0fs）没有转录出任何文本：疑似纯音乐/静音，或切片内容异常",
                    chunk.index, chunk.start_sec, chunk.end_sec,
                )
            free_cuda_memory()

        if not collected:
            raise TranscriptionError("所有切片都没有转录出文本，请检查音频是否含有效语音")

        empty_chunks = [c for c in chunk_stats if c["segments"] == 0]
        if empty_chunks and total_duration:
            covered = sum(c["duration"] for c in empty_chunks)
            logger.warning(
                "%d/%d 个切片没有文本，共 %.0fs（%.1f%%）：%s",
                len(empty_chunks), len(chunk_stats), covered, covered / total_duration * 100,
                [
                    f"{c['start']:.0f}-{c['start'] + c['duration']:.0f}s"
                    for c in empty_chunks
                ],
            )

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

        # 顶层 language 不再随便挑一个：没有明显主导语言时如实写 multilingual
        if detected_languages:
            top, count = Counter(detected_languages).most_common(1)[0]
            share = count / len(detected_languages)
            language_out = top if share >= LANG_VOTE_MIN_SHARE else "multilingual"
        else:
            language_out = lang or "auto"

        speech_sec = round(sum(max(0.0, s["end"] - s["start"]) for s in segments), 3)
        speech_ratio = round(speech_sec / total_duration, 4) if total_duration else 0.0
        if speech_ratio < 0.01:
            logger.warning(
                "有效语音只占 %.2f%%，疑似纯音乐/无人讲解：%s", speech_ratio * 100, source_wav.name
            )

        # 分离覆盖与转录语音量应大致一致：偏差过大说明分离或转录有问题
        if diar_speech_sec > 0:
            delta = (speech_sec - diar_speech_sec) / diar_speech_sec * 100
            logger.info(
                "分离/转录一致性：分离 %.0fs / 转录 %.0fs（偏差 %+.1f%%）",
                diar_speech_sec, speech_sec, delta,
            )
            if abs(delta) > 20:
                logger.warning(
                    "分离与转录的语音量相差 %.1f%%，建议检查 transcript.json 与分离结果",
                    delta,
                )

        result_payload = {
            "language": language_out,
            "chunk_languages": detected_languages,
            "chunks": chunk_stats,
            "segments": segments,
            "speakers": [mapping[k] for k in sorted(mapping, key=lambda k: int(mapping[k].split()[-1]))],
            "speaker_stats": _speaker_stats(turns, mapping),
            "num_segments": len(segments),
            "duration_sec": total_duration,
            "diarized": bool(turns),
            # 分离为什么没成功：写进产物，UI/总结不必再翻日志
            "diarization": {
                "requested": bool(do_diarize),
                "ok": bool(turns),
                "input_mode": DIAR_AUDIO_INPUT,
                "reason": (None if turns else _DIAR_UNAVAILABLE or _DIAR_LAST_ERROR) or None,
            },
            "speech_ratio": speech_ratio,
            "elapsed_sec": round(time.time() - started, 1),
        }
        logger.info(
            "转录完成：%d 段 / %d 位说话人 / 语音占比 %.1f%% / 总耗时 %.1fs",
            result_payload["num_segments"], len(result_payload["speakers"]),
            speech_ratio * 100, result_payload["elapsed_sec"],
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
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> dict:
    """端到端便利函数：媒体文件 -> 转录结果（内部先调 audio.analyze）。"""
    chunks = analyze(media_path, chunk_minutes=chunk_minutes)
    try:
        return transcribe_task(
            chunks,
            language=language,
            do_diarize=do_diarize,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
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
    parser.add_argument("--speakers", type=int, default=None, help="已知说话人数（同时设 min/max）")
    parser.add_argument("--out", default=None, help="结果 JSON 输出路径")
    parser.add_argument("--print", type=int, default=10, help="打印前 N 段")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    payload = transcribe_file(
        args.input,
        chunk_minutes=args.chunk_minutes,
        language=args.language,
        do_diarize=not args.no_diarize,
        min_speakers=args.speakers,
        max_speakers=args.speakers,
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
