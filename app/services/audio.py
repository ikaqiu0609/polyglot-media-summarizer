"""音频预处理模块。

职责（只做音频，不碰 ASR）：
    1. 用 FFmpeg 从任意容器（mp4/mkv/mp3/wav ...）中抽取 16kHz / 单声道 / PCM16 的 WAV；
    2. 按时长切片：超过 CHUNK_MINUTES（默认 15 分钟）就切成多个小文件，
       并记录每片的起始偏移量，供后续把局部时间戳还原成全局时间戳；
    3. 提供 analyze() 一站式入口：给一个视频/音频路径，返回待转录的切片列表。

设计要点：
    - 全部走 subprocess + 参数列表（不经 shell），文件名含空格/中文也安全；
    - -nostdin 防止 FFmpeg 抢占父进程 stdin（FastAPI 后台任务里很关键）；
    - 切片用 -ss/-t + -c:a copy，PCM 直接拷贝不重编码，3 小时音频切片几秒完成；
    - 切片起点由 plan_chunks() 纯函数计算，误差只可能是 PCM packet 对齐量级（~64ms）。
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 可配置项（全部允许用环境变量覆盖）
# ---------------------------------------------------------------------------
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")

SAMPLE_RATE = 16_000                                    # Whisper 系列固定 16k
CHANNELS = 1
CHUNK_MINUTES = float(os.getenv("CHUNK_MINUTES", "15"))  # 单文件最长时长（分钟）
FFMPEG_TIMEOUT = int(os.getenv("FFMPEG_TIMEOUT", "1800"))  # 单次 FFmpeg 调用超时（秒）
MIN_CHUNK_SEC = 1.0                                     # 小于该时长的尾片直接丢弃

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.getenv("DATA_ROOT", str(PROJECT_ROOT / "data")))
AUDIO_DIR = DATA_ROOT / "audio"

SUPPORTED_EXTS = {
    ".mp4", ".mkv", ".mov", ".webm", ".flv", ".avi", ".m4v", ".ts",
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".amr", ".wma",
}


class AudioProcessingError(RuntimeError):
    """音频处理失败（统一异常类型，方便上层翻译成 HTTP 400/500）。"""


@dataclass
class AudioChunk:
    """一个待转录的音频切片。

    start_sec: 该片在**原始音频**中的起始偏移，转录后要加到每个 segment 的时间戳上。
    """

    index: int
    path: Path
    start_sec: float
    duration_sec: float
    sample_rate: int = SAMPLE_RATE
    # 切片所属的「整段 16k wav」。说话人分离只在整段上跑一次，
    # 否则每片独立聚类，SPEAKER_00 在片与片之间并不指同一个人。
    source: Path | None = None
    meta: dict = field(default_factory=dict)

    @property
    def end_sec(self) -> float:
        return round(self.start_sec + self.duration_sec, 3)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "path": str(self.path),
            "start_sec": round(self.start_sec, 3),
            "duration_sec": round(self.duration_sec, 3),
            "end_sec": self.end_sec,
            "sample_rate": self.sample_rate,
        }


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def _tail(text: str | bytes | None, limit: int = 800) -> str:
    """截取 stderr 末尾，避免把几 MB 日志塞进异常信息。"""
    if not text:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    text = text.strip()
    return text[-limit:]


def run_command(cmd: list[str], stage: str, timeout: int | None = FFMPEG_TIMEOUT) -> str:
    """执行外部命令，统一异常与日志。返回 stdout。"""
    logger.debug("[%s] %s", stage, " ".join(cmd))
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:  # 二进制不存在
        raise AudioProcessingError(
            f"未找到可执行文件 {cmd[0]}，请确认镜像已安装 ffmpeg"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioProcessingError(
            f"{stage} 超时（>{timeout}s），输入文件可能过大或已损坏"
        ) from exc

    stdout = (proc.stdout or b"").decode("utf-8", errors="replace")
    stderr = _tail(proc.stderr)

    if proc.returncode != 0:
        logger.error("[%s] ffmpeg 退出码 %s: %s", stage, proc.returncode, stderr)
        raise AudioProcessingError(f"{stage} 失败（退出码 {proc.returncode}）：{stderr}")

    logger.info("[%s] 完成，耗时 %.1fs", stage, time.time() - started)
    return stdout


def check_ffmpeg() -> dict:
    """启动自检：确认 ffmpeg/ffprobe 可用，返回版本信息。"""
    info: dict[str, str] = {}
    for name, binary in (("ffmpeg", FFMPEG_BIN), ("ffprobe", FFPROBE_BIN)):
        path = shutil.which(binary)
        if not path:
            raise AudioProcessingError(f"环境缺少 {name}（{binary}），请重新构建镜像")
        out = run_command([path, "-version"], stage=f"{name}-version", timeout=30)
        info[name] = out.splitlines()[0].strip() if out else "unknown"
    logger.info("ffmpeg 自检通过: %s", info)
    return info


def is_supported_media(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTS


# ---------------------------------------------------------------------------
# 探测
# ---------------------------------------------------------------------------
def probe_duration(path: str | Path) -> float:
    """用 ffprobe 读取时长（秒）。"""
    path = Path(path)
    if not path.exists():
        raise AudioProcessingError(f"文件不存在: {path}")

    out = run_command(
        [
            FFPROBE_BIN, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        stage="ffprobe-duration",
        timeout=120,
    ).strip()
    try:
        duration = float(out)
    except ValueError as exc:
        raise AudioProcessingError(f"无法解析音频时长（ffprobe 输出: {out!r}）") from exc
    if duration <= 0:
        raise AudioProcessingError(f"音频时长异常: {duration}")
    return duration


def has_audio_stream(path: str | Path) -> bool:
    """判断容器内是否存在音频轨道（只有视频的 mp4 要给出明确报错）。"""
    out = run_command(
        [
            FFPROBE_BIN, "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            str(path),
        ],
        stage="ffprobe-audio-stream",
        timeout=120,
    )
    return "audio" in out.lower()


# ---------------------------------------------------------------------------
# 抽取
# ---------------------------------------------------------------------------
def extract_audio(
    src: str | Path,
    dst: str | Path | None = None,
    work_dir: str | Path | None = None,
) -> Path:
    """抽取 16kHz 单声道 PCM16 WAV。

    :param src:      源视频/音频
    :param dst:      目标 wav 路径，None 则自动生成
    :param work_dir: 自动生成目标时的输出目录，默认 data/audio
    :return: 生成的 wav 路径
    """
    src = Path(src)
    if not src.exists() or src.stat().st_size == 0:
        raise AudioProcessingError(f"输入文件不存在或为空: {src}")
    if not has_audio_stream(src):
        raise AudioProcessingError(f"文件中没有音频轨道: {src.name}")

    if dst is None:
        target_dir = Path(work_dir) if work_dir else AUDIO_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        dst = target_dir / f"{src.stem}.{uuid.uuid4().hex[:8]}.16k.wav"
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    tmp = dst.with_name(dst.name + ".part")
    run_command(
        [
            FFMPEG_BIN, "-hide_banner", "-nostdin", "-y",
            "-i", str(src),
            "-map", "0:a:0",          # 只取第一条音轨
            "-vn",                    # 丢掉视频
            "-ac", str(CHANNELS),
            "-ar", str(SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            "-f", "wav",
            str(tmp),
        ],
        stage=f"extract:{src.name}",
    )

    if not tmp.exists() or tmp.stat().st_size <= 44:  # 44 = WAV header
        tmp.unlink(missing_ok=True)
        raise AudioProcessingError(f"抽取音频失败，输出为空: {src.name}")
    tmp.replace(dst)  # 原子替换，避免残留半成品被后续流程读到
    logger.info("音频抽取完成: %s -> %s (%.1f MB)", src.name, dst.name, dst.stat().st_size / 1e6)
    return dst


# ---------------------------------------------------------------------------
# 切片
# ---------------------------------------------------------------------------
def plan_chunks(total_duration: float, chunk_seconds: float) -> list[tuple[int, float, float]]:
    """纯函数：按总时长规划切片，返回 [(index, start_sec, duration_sec), ...]。

    单独拆出来是为了能脱离 ffmpeg 做单元测试。
    """
    if total_duration <= 0:
        return []
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds 必须大于 0")

    count = math.ceil(total_duration / chunk_seconds - 1e-9)
    plans: list[tuple[int, float, float]] = []
    for i in range(count):
        start = round(i * chunk_seconds, 3)
        duration = round(min(chunk_seconds, total_duration - start), 3)
        if duration < MIN_CHUNK_SEC:
            logger.warning("尾片仅 %.2fs，已丢弃（index=%d）", duration, i)
            continue
        plans.append((i, start, duration))
    return plans


def split_audio(
    wav_path: str | Path,
    chunk_seconds: float,
    out_dir: str | Path | None = None,
) -> list[AudioChunk]:
    """把一个 wav 按固定时长切成多片，返回带全局偏移的 AudioChunk 列表。"""
    wav_path = Path(wav_path)
    total = probe_duration(wav_path)
    plans = plan_chunks(total, chunk_seconds)
    logger.info("切分 %s: 总长 %.1fs -> %d 片（每片 %.0fs）", wav_path.name, total, len(plans), chunk_seconds)

    target_dir = Path(out_dir) if out_dir else wav_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[AudioChunk] = []
    created: list[Path] = []
    try:
        for index, start, duration in plans:
            part = target_dir / f"{wav_path.stem}.part{index:03d}.wav"
            run_command(
                [
                    FFMPEG_BIN, "-hide_banner", "-nostdin", "-y",
                    # -ss 放在 -i 之前 = 输入端 seek，快；PCM 配合 copy 仍有毫秒级精度
                    "-ss", f"{start:.3f}",
                    "-t", f"{duration:.3f}",
                    "-i", str(wav_path),
                    "-c:a", "copy",
                    "-f", "wav",
                    str(part),
                ],
                stage=f"split:{wav_path.name}#{index}",
            )
            if not part.exists() or part.stat().st_size <= 44:
                raise AudioProcessingError(f"切片 {index} 生成失败: {part.name}")
            created.append(part)
            chunks.append(
                AudioChunk(
                    index=index,
                    path=part,
                    start_sec=start,
                    duration_sec=duration,
                    sample_rate=SAMPLE_RATE,
                    source=wav_path,
                )
            )
    except Exception:
        for p in created:  # 失败即清理，别留垃圾文件
            Path(p).unlink(missing_ok=True)
        raise
    return chunks


# ---------------------------------------------------------------------------
# 一站式入口
# ---------------------------------------------------------------------------
def analyze(
    src: str | Path,
    work_dir: str | Path | None = None,
    chunk_minutes: float | None = None,
) -> list[AudioChunk]:
    """视频/音频 -> 待转录切片列表。

    - 时长 <= 阈值：返回 1 个切片（start_sec = 0）
    - 时长 >  阈值：切片后返回多个，start_sec 为各片在原始音频中的偏移
    """
    src = Path(src)
    if not is_supported_media(src.name):
        raise AudioProcessingError(
            f"不支持的媒体格式 {src.suffix}，支持: {', '.join(sorted(SUPPORTED_EXTS))}"
        )

    threshold_min = CHUNK_MINUTES if chunk_minutes is None else chunk_minutes
    chunk_seconds = max(threshold_min, 0.1) * 60.0

    work_dir = Path(work_dir) if work_dir else AUDIO_DIR
    wav = extract_audio(src, work_dir=work_dir)
    total = probe_duration(wav)

    if total <= chunk_seconds:
        logger.info("音频 %.1fs 未超过阈值 %.0fs，不切片", total, chunk_seconds)
        return [
            AudioChunk(
                index=0,
                path=wav,
                start_sec=0.0,
                duration_sec=round(total, 3),
                source=wav,
            )
        ]

    return split_audio(wav, chunk_seconds=chunk_seconds, out_dir=work_dir)


def cleanup(chunks: list[AudioChunk], remove_source: bool = False) -> None:
    """删除切片（转录完成后释放磁盘；整段 wav 可保留）。"""
    for chunk in chunks:
        if remove_source:
            Path(chunk.path).unlink(missing_ok=True)
        elif chunk.index > 0:
            Path(chunk.path).unlink(missing_ok=True)


def cleanup_task_dir(work_dir: str | Path) -> None:
    """删除整个任务音频目录（任务收尾时调用）。"""
    path = Path(work_dir)
    if path.exists() and path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        logger.info("已清理任务音频目录: %s", path)


# ---------------------------------------------------------------------------
# CLI：python -m app.services.audio <file> [--chunk-minutes 15]
# ---------------------------------------------------------------------------
def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="FFmpeg 抽取 + 切片")
    parser.add_argument("input", help="视频或音频文件路径")
    parser.add_argument("--chunk-minutes", type=float, default=None, help="切片阈值（分钟）")
    parser.add_argument("--work-dir", default=None, help="输出目录")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    chunks = analyze(args.input, work_dir=args.work_dir, chunk_minutes=args.chunk_minutes)
    print(json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
