"""重叠语音分离（主进程侧）。

分离模型只能跑在 pyannote.audio 3.3.2，而主服务是 4.x，两者不能共存。
因此这里只做三件事：
    1. 从 diarization turns 里算出「两个人以上同时说话」的区间；
    2. 用 subprocess 调独立 venv 里的 scripts/separate_overlap.py；
    3. 读取它写出的 manifest（每条轨的 wav 路径、说话人、时间）。

真正把分离轨喂给 ASR 的逻辑在 transcribe.py（需要 ASR 模型句柄）。
任何一步失败都返回 None，调用方回退到原转录结果，不阻塞主流程。
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

from app.config import PROJECT_ROOT, Settings

logger = logging.getLogger(__name__)

SEPARATION_SCRIPT = PROJECT_ROOT / "scripts" / "separate_overlap.py"


def overlap_spans(
    turns: Sequence[Any],
    min_sec: float = 0.8,
    join_gap: float = 0.5,
) -> list[tuple[float, float]]:
    """从说话人时间片里算出「≥2 人同时说话」的区间。

    纯函数，方便单测。turns 只需要有 start / end 两个属性（DiarTurn 即可）。

    :param min_sec:  短于该时长的重叠忽略（换人瞬间常有一两帧误判）
    :param join_gap: 相邻重叠区间间隔小于该值时合并
    """
    events: list[tuple[float, int]] = []
    for turn in turns:
        start = float(getattr(turn, "start", 0.0))
        end = float(getattr(turn, "end", start))
        if end <= start:
            continue
        events.append((start, 1))
        events.append((end, -1))
    if not events:
        return []

    # 同一时间点先处理结束（-1），避免「上一段结束、下一段开始」被当成重叠
    events.sort(key=lambda item: (item[0], item[1]))

    spans: list[tuple[float, float]] = []
    active = 0
    start_at: float | None = None
    for time, delta in events:
        active += delta
        if delta > 0 and active >= 2 and start_at is None:
            start_at = time
        elif delta < 0 and active < 2 and start_at is not None:
            spans.append((start_at, time))
            start_at = None

    merged: list[list[float]] = []
    for start, end in spans:
        if merged and start - merged[-1][1] <= join_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged if end - start >= min_sec]


def available(settings: Settings) -> bool:
    """分离环境是否就绪（不实际加载模型）。"""
    if not settings.separation_enabled:
        return False
    if not Path(settings.separation_python).exists():
        return False
    return SEPARATION_SCRIPT.exists()


def separate(
    wav_path: str | Path,
    start: float,
    end: float,
    out_dir: str | Path,
    settings: Settings,
) -> dict[str, Any] | None:
    """调用独立 venv 做一次分离，返回 manifest；失败返回 None。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest_path.unlink(missing_ok=True)

    cmd = [
        settings.separation_python,
        str(SEPARATION_SCRIPT),
        "--wav", str(wav_path),
        "--start", f"{start:.3f}",
        "--end", f"{end:.3f}",
        "--out-dir", str(out_dir),
        "--manifest", str(manifest_path),
        "--model", settings.separation_model,
    ]
    logger.info(
        "重叠段分离：%.1f~%.1fs（%.1fs），模型 %s",
        start, end, max(0.0, end - start), settings.separation_model,
    )
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=settings.separation_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning("重叠段分离超时（>%ss），保留原转录结果", settings.separation_timeout)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("无法启动分离子进程（%s），保留原转录结果", exc)
        return None

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-600:]
        logger.warning(
            "重叠段分离失败（退出码 %s）：%s；保留原转录结果", proc.returncode, tail
        )
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("分离 manifest 读取失败（%s），保留原转录结果", exc)
        return None

    manifest["out_dir"] = str(out_dir)
    tracks = manifest.get("tracks") or []
    logger.info(
        "重叠段分离完成 %.1fs：%d 轨（%s）",
        time.time() - started,
        len(tracks),
        [track.get("speaker") for track in tracks],
    )
    return manifest
