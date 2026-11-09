"""FastAPI 主应用：上传 -> 转录（免费）-> 用户确认 -> 总结（付费）-> 下载。

两段式设计（AUTO_SUMMARIZE=false，默认）：
    转录完全在本地 GPU 上完成，不产生任何 API 费用；产物
    data/outputs/{task_id}.transcript.json 落盘后任务停在 `transcribed`。
    只有用户显式调用 /summarize/* 才会调用 DeepSeek 并计费。
    设 AUTO_SUMMARIZE=true 可恢复「转录完立即总结」的旧行为。

接口：
    POST /upload              上传媒体文件，立即返回 task_id（后台抽取+转录）
    POST /summarize/{task_id} 对已转录任务触发总结（会调用 DeepSeek，产生费用）
    POST /summarize           导入现成的 transcript.json 并总结
    GET  /status/{task_id}    查询状态与进度百分比
    GET  /result/{task_id}    下载 Markdown 总结
    GET  /healthz             配置与依赖自检（不返回任何密钥）

并发模型：
    BackgroundTasks 跑在线程池里，函数是同步阻塞的（subprocess + GPU 推理）。
    由于显存独占，用 Semaphore(MAX_CONCURRENT_TASKS，默认 1) 串行化，
    排队的任务状态保持 queued，进度为 0。
"""

from __future__ import annotations

import importlib.util
import json
import logging
import shutil
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import settings
from app.services import audio, summarizer, transcribe

logger = logging.getLogger("app.main")


# ---------------------------------------------------------------------------
# 状态机
# ---------------------------------------------------------------------------
class TaskStatus(str, Enum):
    QUEUED = "queued"
    EXTRACTING = "extracting"
    DIARIZING = "diarizing"
    TRANSCRIBING = "transcribing"
    # 转录完成、产物已落盘，但还没调用付费的总结 API。
    # 这是「等待用户决定」的中间状态，不是失败，也不是完成。
    TRANSCRIBED = "transcribed"
    SUMMARIZING = "summarizing"
    COMPLETED = "completed"
    FAILED = "failed"


STAGE_LABELS: dict[TaskStatus, str] = {
    TaskStatus.QUEUED: "排队中",
    TaskStatus.EXTRACTING: "抽取音频",
    TaskStatus.DIARIZING: "说话人分离",
    TaskStatus.TRANSCRIBING: "语音转录",
    TaskStatus.TRANSCRIBED: "转录完成（待总结）",
    TaskStatus.SUMMARIZING: "生成总结",
    TaskStatus.COMPLETED: "已完成",
    TaskStatus.FAILED: "失败",
}

# 各阶段在总进度中的占比区间（%），由 transcribe 的进度回调驱动
STAGE_RANGES: dict[TaskStatus, tuple[float, float]] = {
    TaskStatus.EXTRACTING: (2.0, 15.0),
    TaskStatus.DIARIZING: (15.0, 35.0),
    TaskStatus.TRANSCRIBING: (35.0, 75.0),
    TaskStatus.SUMMARIZING: (75.0, 98.0),
}


# ---------------------------------------------------------------------------
# 任务记录（内存态，进程重启即丢失；如需持久化可换 Redis/DB）
# ---------------------------------------------------------------------------
@dataclass
class TaskRecord:
    task_id: str
    filename: str
    source_path: Path
    size_bytes: int
    status: TaskStatus = TaskStatus.QUEUED
    progress: float = 0.0
    detail: str = "等待处理"
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    summary_path: Path | None = None
    result: dict[str, Any] | None = None
    # 上传时由用户给出的已知说话人数（None = 交给 pyannote 自动判断）
    speaker_count: int | None = None

    @property
    def elapsed_sec(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or datetime.now(timezone.utc)
        return round((end - self.started_at).total_seconds(), 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "stage": STAGE_LABELS.get(self.status, self.status.value),
            "progress": round(self.progress, 1),
            "detail": self.detail,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat(timespec="seconds"),
            "elapsed_sec": self.elapsed_sec,
            "error": self.error,
            "result": self.result,
        }


_tasks: dict[str, TaskRecord] = {}
_tasks_lock = threading.RLock()
_pipeline_semaphore = threading.Semaphore(settings.max_concurrent_tasks)

# 导入 transcript.json 的大小上限（1 小时转录 JSON 通常只有几百 KB）
MAX_TRANSCRIPT_MB = 50
_MAX_TRANSCRIPT_BYTES = MAX_TRANSCRIPT_MB * 1024 * 1024

_UPDATABLE = {
    "status", "progress", "detail", "error",
    "started_at", "finished_at", "summary_path", "result",
}


def _register(record: TaskRecord) -> None:
    with _tasks_lock:
        _tasks[record.task_id] = record


def _get(task_id: str) -> TaskRecord | None:
    with _tasks_lock:
        return _tasks.get(task_id)


_TERMINAL_STATUSES = {TaskStatus.COMPLETED, TaskStatus.FAILED}
# 这些状态只能向前推进，不能被迟到的进度回调拉回去
_EARLY_STATUSES = {
    TaskStatus.QUEUED,
    TaskStatus.EXTRACTING,
    TaskStatus.DIARIZING,
    TaskStatus.TRANSCRIBING,
}


def _update(task_id: str, *, force: bool = False, **changes: Any) -> None:
    with _tasks_lock:
        record = _tasks.get(task_id)
        if record is None:
            return
        new_status = changes.get("status")
        if not force and new_status is not None:
            # 终态不可回退（后台线程迟到的一次进度回调不应把 completed 改回 transcribing）
            if record.status in _TERMINAL_STATUSES and new_status not in _TERMINAL_STATUSES:
                logger.debug("任务 %s 已终态（%s），忽略状态回退", task_id, record.status.value)
                return
            # 转录完成后在等用户触发总结：忽略转录阶段的迟到进度回调
            if record.status == TaskStatus.TRANSCRIBED and new_status in _EARLY_STATUSES:
                logger.debug("任务 %s 等待总结，忽略迟到的进度（%s）", task_id, new_status.value)
                return
        for key, value in changes.items():
            if key in _UPDATABLE:
                setattr(record, key, value)
        record.progress = max(0.0, min(100.0, float(record.progress)))


def _purge_expired() -> None:
    """清理超过保留期的任务，防止内存与磁盘无限增长。"""
    if settings.task_retention_hours <= 0:
        return
    deadline = time.time() - settings.task_retention_hours * 3600
    with _tasks_lock:
        expired = [
            task_id
            for task_id, record in _tasks.items()
            if (record.finished_at or record.created_at).timestamp() < deadline
            # transcribed 也在等保留期：用户一直不总结的话，产物同样会被回收
            and record.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.TRANSCRIBED}
        ]
        for task_id in expired:
            _tasks.pop(task_id, None)
    for task_id in expired:
        for path in (
            settings.upload_dir.glob(f"{task_id}.*"),
            [settings.summary_path(task_id), settings.transcript_path(task_id)],
        ):
            for item in (path if isinstance(path, list) else list(path)):
                Path(item).unlink(missing_ok=True)
        logger.info("已清理过期任务 %s", task_id)


# ---------------------------------------------------------------------------
# 编排：audio -> transcribe -> summarizer
# ---------------------------------------------------------------------------
def _progress_bridge(task_id: str, default_stage: TaskStatus) -> Callable[[str, float, str], None]:
    """把 transcribe 的阶段内进度（0~1）映射成整体进度（%）。"""

    def callback(stage: str, fraction: float, detail: str = "") -> None:
        try:
            stage_status = TaskStatus(stage)
        except ValueError:
            stage_status = default_stage
        low, high = STAGE_RANGES.get(stage_status, (0.0, 100.0))
        percent = low + (high - low) * max(0.0, min(1.0, fraction))
        _update(
            task_id,
            status=stage_status,
            progress=percent,
            detail=detail or STAGE_LABELS.get(stage_status, stage_status.value),
        )

    return callback


def _acquire_pipeline_slot(task_id: str) -> None:
    """串行化 GPU 任务；排队时把 detail 写清楚，避免 /status 看起来像卡死。"""
    if _pipeline_semaphore.acquire(blocking=False):
        return
    _update(task_id, detail="排队中：等待前一个任务释放 GPU")
    _pipeline_semaphore.acquire()


def _summary_result(
    task_id: str,
    transcribed: dict,
    transcript_path: Path,
    total_audio: float,
    summary: dict,
) -> dict[str, Any]:
    return {
        "summary_path": str(settings.summary_path(task_id)),
        "transcript_path": str(transcript_path),
        "num_segments": transcribed.get("num_segments", 0),
        "num_blocks": summary["meta"].get("num_blocks", 0),
        "speakers": transcribed.get("speakers", []),
        "speaker_stats": transcribed.get("speaker_stats", []),
        "language": transcribed.get("language"),
        "diarized": bool(summary["meta"].get("diarized")),
        "speaker_mode": summary["meta"].get("speaker_mode"),
        "diarization": transcribed.get("diarization"),
        "audio_duration_sec": total_audio,
        "speech_ratio": transcribed.get("speech_ratio"),
        "tokens": summary["usage"],
    }


def _transcribed_result(
    task_id: str,
    transcribed: dict,
    transcript_path: Path,
    total_audio: float,
) -> dict[str, Any]:
    """转录完成但还没总结时对外暴露的信息（进程重启后重建任务也复用它）。"""
    return {
        "transcript_path": str(transcript_path),
        "num_segments": transcribed.get("num_segments", 0),
        "speakers": transcribed.get("speakers", []),
        "speaker_stats": transcribed.get("speaker_stats", []),
        "language": transcribed.get("language"),
        "diarized": bool(transcribed.get("diarized")),
        "diarization": transcribed.get("diarization"),
        "audio_duration_sec": total_audio,
        "speech_ratio": transcribed.get("speech_ratio"),
        # 哪些切片完全没出文本（纯音乐/静音），便于排查「转录结果缺一段」
        "empty_chunks": [
            c.get("start") for c in (transcribed.get("chunks") or []) if not c.get("segments")
        ],
        "summarize_url": f"/summarize/{task_id}",
        "hint": (
            "转录免费且已完成。需要总结时 POST /summarize/{task_id}"
            "（会调用 DeepSeek 并产生费用）；想自动执行请设 AUTO_SUMMARIZE=true。"
        ),
    }


def _summarize_and_complete(
    task_id: str,
    transcribed: dict,
    transcript_path: Path,
    total_audio: float,
) -> None:
    """调用 DeepSeek 生成总结并置为 completed（仅 AUTO_SUMMARIZE=true 时走这里）。"""
    _update(
        task_id,
        status=TaskStatus.SUMMARIZING,
        progress=STAGE_RANGES[TaskStatus.SUMMARIZING][0],
        detail=f"DeepSeek 分段摘要（{transcribed.get('num_segments', 0)} 段）",
        error=None,
    )
    summary = summarizer.summarize(
        transcribed,
        block_chars=settings.summary_block_chars,
        save_to=settings.summary_path(task_id),
    )
    _update(
        task_id,
        status=TaskStatus.COMPLETED,
        progress=100.0,
        detail="已完成",
        finished_at=datetime.now(timezone.utc),
        summary_path=settings.summary_path(task_id),
        result=_summary_result(task_id, transcribed, transcript_path, total_audio, summary),
    )
    logger.info("任务 %s 完成：%s", task_id, settings.summary_path(task_id))


def _pause_for_summary(
    task_id: str,
    transcribed: dict,
    transcript_path: Path,
    total_audio: float,
) -> None:
    """转录已完成但未调用付费 API：停在 transcribed，等用户显式触发。"""
    _update(
        task_id,
        status=TaskStatus.TRANSCRIBED,
        progress=STAGE_RANGES[TaskStatus.TRANSCRIBING][1],
        detail="转录完成，等待手动触发总结（未调用付费 API）",
        finished_at=datetime.now(timezone.utc),
        result=_transcribed_result(task_id, transcribed, transcript_path, total_audio),
    )
    logger.info("任务 %s 转录完成，等待手动触发总结：%s", task_id, transcript_path)


def run_pipeline(task_id: str, speaker_count: int | None = None) -> None:
    """后台任务主体（同步阻塞，由 BackgroundTasks 放到线程池执行）。

    只负责「抽取 → 转录 → 落盘 JSON」。总结默认**不自动执行**：
    DeepSeek 调用要花钱，由用户通过 POST /summarize/{task_id} 显式决定。

    :param speaker_count: 用户声明的已知说话人数（同时作为 min/max 传给 pyannote）
    """
    record = _get(task_id)
    if record is None:
        logger.error("任务 %s 不存在，跳过", task_id)
        return

    _acquire_pipeline_slot(task_id)

    work_dir = settings.task_audio_dir(task_id)
    try:
        _update(
            task_id,
            status=TaskStatus.EXTRACTING,
            progress=STAGE_RANGES[TaskStatus.EXTRACTING][0],
            detail="FFmpeg 抽取 16kHz 单声道音频",
            started_at=datetime.now(timezone.utc),
        )

        # ---- 1. 音频抽取与切片 ----
        chunks = audio.analyze(
            record.source_path,
            work_dir=work_dir,
            chunk_minutes=settings.chunk_minutes,
        )
        total_audio = round(sum(c.duration_sec for c in chunks), 2)
        _update(
            task_id,
            progress=STAGE_RANGES[TaskStatus.EXTRACTING][1],
            detail=f"音频 {total_audio:.0f}s，切分为 {len(chunks)} 片",
        )

        # ---- 2. 转录 + 说话人分离（纯本地，不产生 API 费用）----
        transcribed = transcribe.transcribe_task(
            chunks,
            language=settings.asr_language,
            do_diarize=settings.enable_diarization,
            progress=_progress_bridge(task_id, TaskStatus.TRANSCRIBING),
            min_speakers=speaker_count,
            max_speakers=speaker_count,
        )
        # 落盘时带上来源信息：进程重启后靠它把 outputs 里的产物还原成任务
        payload = dict(transcribed)
        payload["source_filename"] = record.filename
        payload["source_size_bytes"] = record.size_bytes
        transcript_path = settings.transcript_path(task_id)
        transcript_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _update(
            task_id,
            progress=STAGE_RANGES[TaskStatus.TRANSCRIBING][1],
            detail="转录完成，JSON 已落盘",
        )

        # ---- 3. 收尾：自动总结 or 停在这里等用户确认 ----
        if settings.auto_summarize:
            _summarize_and_complete(task_id, transcribed, transcript_path, total_audio)
        else:
            _pause_for_summary(task_id, transcribed, transcript_path, total_audio)

    except audio.AudioProcessingError as exc:
        _fail(task_id, f"音频处理失败：{exc}")
    except transcribe.TranscriptionError as exc:
        _fail(task_id, f"转录失败：{exc}")
    except summarizer.SummarizationError as exc:
        _fail(task_id, f"总结失败：{exc}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("任务 %s 出现未预期异常", task_id)
        _fail(task_id, f"处理异常：{exc}")
    finally:
        # 释放切片/整段 wav（原始上传文件保留，便于重跑或排查）
        audio.cleanup_task_dir(work_dir)
        _pipeline_semaphore.release()


def run_summarize(task_id: str) -> None:
    """只跑总结：读取已落盘的 {task_id}.transcript.json，不重新转录、不重新上传。

    供 POST /summarize/{task_id} 与 POST /summarize（导入 JSON）复用，
    也是总结阶段失败（如 402 余额不足）后的恢复路径。
    """
    record = _get(task_id)
    if record is None:
        logger.error("任务 %s 不存在，跳过总结", task_id)
        return

    transcript_path = settings.transcript_path(task_id)
    if not transcript_path.exists():
        _fail(task_id, f"找不到转录产物：{transcript_path}（请重新上传媒体或导入 transcript.json）")
        return

    _acquire_pipeline_slot(task_id)
    try:
        payload = json.loads(transcript_path.read_text(encoding="utf-8"))
        _update(
            task_id,
            force=True,
            status=TaskStatus.SUMMARIZING,
            progress=STAGE_RANGES[TaskStatus.SUMMARIZING][0],
            detail=f"DeepSeek 分段摘要（{payload.get('num_segments', 0)} 段）",
            error=None,
            finished_at=None,
        )
        summary = summarizer.summarize(
            payload,
            block_chars=settings.summary_block_chars,
            save_to=settings.summary_path(task_id),
        )
        _update(
            task_id,
            force=True,
            status=TaskStatus.COMPLETED,
            progress=100.0,
            detail="已完成",
            finished_at=datetime.now(timezone.utc),
            summary_path=settings.summary_path(task_id),
            result=_summary_result(
                task_id, payload, transcript_path, float(payload.get("duration_sec") or 0.0), summary
            ),
        )
        logger.info("任务 %s 总结完成：%s", task_id, settings.summary_path(task_id))
    except summarizer.SummarizationError as exc:
        _fail(task_id, f"总结失败：{exc}")
    except json.JSONDecodeError as exc:
        _fail(task_id, f"转录 JSON 解析失败（{transcript_path}）：{exc}")
    except Exception as exc:  # noqa: BLE001
        logger.exception("任务 %s 总结时出现未预期异常", task_id)
        _fail(task_id, f"总结异常：{exc}")
    finally:
        _pipeline_semaphore.release()


def _fail(task_id: str, message: str) -> None:
    logger.error("任务 %s 失败：%s", task_id, message)
    # 转录已完成、只在总结阶段失败时，把产物路径与重跑入口回传，不需要重新上传/转录
    result = None
    transcript = settings.transcript_path(task_id)
    if transcript.exists():
        result = {
            "transcript_path": str(transcript),
            "retry_url": f"/summarize/{task_id}",
            "hint": (
                "转录已完成，修复问题后 POST /summarize/{task_id} 可直接重跑总结"
                "（不会重新转录，也不需要重新上传）。"
            ),
        }
    else:
        # 转录阶段就失败（模型加载、网络、显存）：上传文件还留着，可以直接重跑
        record = _get(task_id)
        # 导入转录 JSON 的任务 source_path 就是 transcript 本身，没有媒体可重跑
        if record is not None and record.source_path.exists() and record.source_path != transcript:
            result = {
                "retry_url": f"/retry/{task_id}",
                "hint": (
                    "原始上传文件仍在。修复环境（网络、模型缓存、显存等）后 "
                    "POST /retry/{task_id} 可直接重跑，无需重新上传。"
                ),
            }
    _update(
        task_id,
        force=True,
        status=TaskStatus.FAILED,
        detail="处理失败",
        error=message[:2000],
        finished_at=datetime.now(timezone.utc),
        result=result,
    )


def _restore_tasks_from_outputs() -> int:
    """进程重启后从 data/outputs 重建任务表。

    转录/总结产物都在磁盘上，但任务表只在内存里 —— 不重建的话，
    重启后 /summarize/{task_id} 和 /result/{task_id} 会直接 404。
    """
    restored = 0
    for path in sorted(settings.output_dir.glob("*.transcript.json")):
        task_id = path.name.split(".")[0]
        if len(task_id) != 32 or _get(task_id) is not None:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("跳过无法解析的转录产物 %s：%s", path.name, exc)
            continue

        finished = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        has_summary = settings.summary_path(task_id).exists()
        filename = str(payload.get("source_filename") or path.name)
        uploaded = settings.upload_path(task_id, Path(filename).suffix.lower())
        record = TaskRecord(
            task_id=task_id,
            filename=filename,
            source_path=uploaded if uploaded.exists() else path,
            size_bytes=int(payload.get("source_size_bytes") or 0),
            status=TaskStatus.COMPLETED if has_summary else TaskStatus.TRANSCRIBED,
            progress=100.0 if has_summary else STAGE_RANGES[TaskStatus.TRANSCRIBING][1],
            detail="已完成" if has_summary else "转录完成，等待手动触发总结（服务重启后恢复）",
            created_at=finished,
            finished_at=finished,
        )
        result = _transcribed_result(
            task_id, payload, path, float(payload.get("duration_sec") or 0.0)
        )
        if has_summary:
            record.summary_path = settings.summary_path(task_id)
            result["summary_path"] = str(record.summary_path)
        record.result = result
        _register(record)
        restored += 1

    if restored:
        logger.info("从 %s 恢复 %d 个历史任务", settings.output_dir, restored)
    return restored


def _restore_orphan_uploads(limit: int = 20) -> int:
    """把 uploads 里「有文件但没有任务记录」的媒体恢复成 failed 任务。

    典型场景：模型加载失败（网络/显存）后重启服务 —— 任务表在内存里丢了，
    但 82MB 的上传文件还在。恢复后可以直接 `POST /retry/{task_id}`，
    不必重新上传。
    """
    if not settings.upload_dir.exists():
        return 0

    deadline = time.time() - max(settings.task_retention_hours, 0) * 3600
    candidates: list[tuple[float, str, Path]] = []
    for path in settings.upload_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in settings.allowed_extensions:
            continue
        task_id = path.name.split(".")[0]
        if len(task_id) != 32 or _get(task_id) is not None:
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if settings.task_retention_hours > 0 and mtime < deadline:
            continue
        candidates.append((mtime, task_id, path))

    restored = 0
    # 只恢复最近的若干个，避免历史残留把任务表撑满
    for mtime, task_id, path in sorted(candidates, reverse=True)[:limit]:
        uploaded_at = datetime.fromtimestamp(mtime, tz=timezone.utc)
        record = TaskRecord(
            task_id=task_id,
            filename=path.name,
            source_path=path,
            size_bytes=path.stat().st_size,
            status=TaskStatus.FAILED,
            detail="服务重启，任务中断",
            error="服务重启导致任务中断；可 POST /retry/{task_id} 重跑（无需重新上传）",
            created_at=uploaded_at,
            finished_at=uploaded_at,
        )
        record.result = {
            "retry_url": f"/retry/{task_id}",
            "hint": "原始上传文件仍在，POST /retry/{task_id} 可重跑转录。",
        }
        _register(record)
        restored += 1

    if restored:
        logger.info("从 uploads 恢复 %d 个未完成任务（可直接 /retry）", restored)
    return restored


# ---------------------------------------------------------------------------
# 上传
# ---------------------------------------------------------------------------
async def _save_upload(upload: UploadFile, dest: Path, max_bytes: int) -> int:
    """流式落盘（不把整个文件读进内存），超限即中断并清理。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with dest.open("wb") as handle:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件超过上限 {settings.max_upload_mb} MB",
                    )
                handle.write(chunk)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="上传内容为空")
    return size


# ---------------------------------------------------------------------------
# 响应模型
# ---------------------------------------------------------------------------
class UploadResponse(BaseModel):
    task_id: str
    status: TaskStatus
    filename: str
    size_bytes: int
    status_url: str
    result_url: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    stage: str
    progress: float
    detail: str
    filename: str
    size_bytes: int
    created_at: str
    elapsed_sec: float
    error: str | None = None
    result: dict[str, Any] | None = None


class SummarizeResponse(BaseModel):
    task_id: str
    status: TaskStatus
    transcript_path: str
    status_url: str
    result_url: str
    note: str


# ---------------------------------------------------------------------------
# 应用
# ---------------------------------------------------------------------------
# 第三方库的 INFO 会刷几百行（每个 HF 请求一条），会淹没业务日志
_NOISY_LOGGERS = (
    "httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3",
    "openai", "filelock", "fsspec", "pyannote",
)


def _module_available(name: str) -> bool:
    """只查包是否存在（不 import），用于 /healthz 的依赖自检。"""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    settings.ensure_dirs()
    logger.info("启动配置：%s", json.dumps(settings.as_dict(), ensure_ascii=False))
    _restore_tasks_from_outputs()
    _restore_orphan_uploads()
    if not shutil.which("ffmpeg"):
        logger.error("环境中找不到 ffmpeg，音频抽取会失败")
    if settings.enable_diarization and not settings.hf_token_present:
        logger.warning("未检测到 HF_TOKEN，任务将降级为「无说话人标签」")
    if settings.enable_diarization:
        logger.info(
            "说话人分离要求同一 HF 账号已 Agree 三个 gated 模型：%s、"
            "pyannote/segmentation-3.0、pyannote/speaker-diarization-community-1（详见 README）",
            settings.diarization_model,
        )
        if transcribe.diarization_status()["input_mode"] == "path" and not _module_available("torchcodec"):
            logger.warning(
                "DIAR_AUDIO_INPUT=path 但 torchcodec 不可用，分离会失败；"
                "改回 waveform 或按 README 安装与 torch 版本匹配的 torchcodec"
            )
    if not settings.deepseek_key_present:
        logger.warning("未检测到 DEEPSEEK_API_KEY，手动总结会失败")
    if not settings.auto_summarize:
        logger.info(
            "AUTO_SUMMARIZE=false：转录完成后停在 transcribed（不调用付费 API），"
            "需显式 POST /summarize/{task_id} 或 POST /summarize 导入 JSON"
        )
    yield
    logger.info("服务关闭")


app = FastAPI(
    title="Polyglot Media Summarizer",
    description="本地 ASR + 说话人分离 + DeepSeek 结构化总结",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/upload", response_model=UploadResponse, status_code=202, summary="上传媒体并排队处理")
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    speakers: int | None = Form(None, ge=1, le=20, description="已知说话人数，留空自动判断"),
) -> UploadResponse:
    _purge_expired()

    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"不支持的格式 {suffix or '(无扩展名)'}，支持：{', '.join(sorted(settings.allowed_extensions))}",
        )

    task_id = uuid.uuid4().hex
    dest = settings.upload_path(task_id, suffix)
    size = await _save_upload(file, dest, settings.max_upload_bytes)
    await file.close()

    _register(
        TaskRecord(
            task_id=task_id,
            filename=filename,
            source_path=dest,
            size_bytes=size,
            speaker_count=speakers,
        )
    )
    logger.info(
        "接收上传 %s（%.1f MB）-> 任务 %s（说话人先验=%s）",
        filename, size / 1e6, task_id, speakers or "自动",
    )

    # BackgroundTasks 会在响应返回后执行，run_pipeline 是同步函数，自动进线程池
    background_tasks.add_task(run_pipeline, task_id, speakers)

    return UploadResponse(
        task_id=task_id,
        status=TaskStatus.QUEUED,
        filename=filename,
        size_bytes=size,
        status_url=f"/status/{task_id}",
        result_url=f"/result/{task_id}",
    )


@app.post(
    "/retry/{task_id}",
    response_model=UploadResponse,
    status_code=202,
    summary="用已上传的文件重跑转录（无需重新上传）",
)
def retry_task(task_id: str, background_tasks: BackgroundTasks) -> UploadResponse:
    """模型加载失败 / 网络抖动 / 显存不足后，重跑同一个任务。

    只适用于「原始上传文件还在」的任务（`/upload` 创建的）。导入转录 JSON 的任务
    没有媒体文件，只能走 `/summarize`。
    """
    record = _get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"任务不存在：{task_id}")
    if record.status not in {TaskStatus.FAILED, TaskStatus.TRANSCRIBED, TaskStatus.COMPLETED}:
        raise HTTPException(
            status_code=409,
            detail=f"任务正在处理中（{record.status.value}），请等结束后再重跑",
        )
    if record.source_path == settings.transcript_path(task_id):
        raise HTTPException(
            status_code=409, detail="该任务来自导入的转录 JSON，没有原始媒体，无法重跑转录"
        )
    if not record.source_path.exists():
        raise HTTPException(
            status_code=410, detail="原始上传文件已被清理（超过保留期），请重新上传"
        )

    _update(
        task_id,
        force=True,
        status=TaskStatus.QUEUED,
        progress=0.0,
        detail="已重新排队",
        error=None,
        finished_at=None,
        summary_path=None,
        result=None,
    )
    background_tasks.add_task(run_pipeline, task_id, record.speaker_count)
    logger.info("任务 %s 收到重跑请求（%s）", task_id, record.filename)
    return UploadResponse(
        task_id=task_id,
        status=TaskStatus.QUEUED,
        filename=record.filename,
        size_bytes=record.size_bytes,
        status_url=f"/status/{task_id}",
        result_url=f"/result/{task_id}",
    )


@app.get("/status/{task_id}", response_model=TaskStatusResponse, summary="查询任务状态")
def get_status(task_id: str) -> TaskStatusResponse:
    record = _get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"任务不存在：{task_id}")
    return TaskStatusResponse(**record.to_dict())


@app.get("/result/{task_id}", summary="下载 Markdown 总结")
def get_result(task_id: str) -> FileResponse:
    record = _get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"任务不存在：{task_id}")
    if record.status == TaskStatus.TRANSCRIBED:
        raise HTTPException(
            status_code=409,
            detail=(
                "转录已完成，但尚未生成总结（避免自动产生 API 费用）。"
                f"请先 POST /summarize/{task_id} 触发总结"
            ),
        )
    if record.status != TaskStatus.COMPLETED:
        raise HTTPException(status_code=409, detail=f"任务尚未完成（当前状态：{record.status.value}）")

    path = record.summary_path or settings.summary_path(task_id)
    if not Path(path).exists():
        raise HTTPException(status_code=410, detail="总结文件已被清理")
    return FileResponse(
        path,
        media_type="text/markdown; charset=utf-8",
        filename=f"summary-{task_id}.md",
    )


# ---------------------------------------------------------------------------
# 手动总结：把「免费转录」与「付费总结」解耦
# ---------------------------------------------------------------------------
def _ensure_summarizable(task_id: str) -> Path:
    """校验任务能否进入总结阶段，返回 transcript 路径。"""
    record = _get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"任务不存在：{task_id}")

    transcript = settings.transcript_path(task_id)
    if not transcript.exists():
        raise HTTPException(status_code=409, detail="该任务没有转录产物，请先上传媒体完成转录")

    if record.status == TaskStatus.SUMMARIZING:
        raise HTTPException(status_code=409, detail="该任务正在总结中，请勿重复触发")
    if record.status in _EARLY_STATUSES:
        raise HTTPException(
            status_code=409, detail=f"任务尚未转录完成（当前状态：{record.status.value}）"
        )
    if record.status == TaskStatus.COMPLETED and settings.summary_path(task_id).exists():
        raise HTTPException(
            status_code=409,
            detail="总结已存在。如需重跑请先删除对应 .md，或改用 POST /summarize 导入 JSON 生成新任务",
        )
    return transcript


@app.post(
    "/summarize/{task_id}",
    response_model=SummarizeResponse,
    status_code=202,
    summary="对已转录任务触发总结（会调用付费 API）",
)
def trigger_summarize(task_id: str, background_tasks: BackgroundTasks) -> SummarizeResponse:
    """转录完成后由用户显式触发；只读 transcript.json，不重新转录。

    幂等：已经在总结中时直接返回 202（而不是 409），避免前端重试把日志刷满。
    """
    record = _get(task_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"任务不存在：{task_id}")
    if record.status == TaskStatus.SUMMARIZING:
        logger.info("任务 %s 已在总结中，忽略重复触发", task_id)
        return SummarizeResponse(
            task_id=task_id,
            status=TaskStatus.SUMMARIZING,
            transcript_path=str(settings.transcript_path(task_id)),
            status_url=f"/status/{task_id}",
            result_url=f"/result/{task_id}",
            note="该任务正在总结中，本次请求未重复触发（不会重复计费）",
        )

    transcript = _ensure_summarizable(task_id)
    _update(
        task_id,
        force=True,
        status=TaskStatus.SUMMARIZING,
        progress=STAGE_RANGES[TaskStatus.SUMMARIZING][0],
        detail="已收到总结请求，排队中",
        error=None,
        finished_at=None,
    )
    background_tasks.add_task(run_summarize, task_id)
    logger.info("任务 %s 收到手动总结请求（将调用 DeepSeek）", task_id)
    return SummarizeResponse(
        task_id=task_id,
        status=TaskStatus.SUMMARIZING,
        transcript_path=str(transcript),
        status_url=f"/status/{task_id}",
        result_url=f"/result/{task_id}",
        note="已开始总结，会调用 DeepSeek API 并产生费用",
    )


@app.post(
    "/summarize",
    response_model=SummarizeResponse,
    status_code=202,
    summary="导入 transcript.json 并生成总结（会调用付费 API）",
)
async def summarize_import(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> SummarizeResponse:
    """把现成的 transcript.json 交给服务总结，适合换机器或复用历史转录。"""
    raw = await file.read(_MAX_TRANSCRIPT_BYTES + 1)
    await file.close()
    if not raw:
        raise HTTPException(status_code=400, detail="上传内容为空")
    if len(raw) > _MAX_TRANSCRIPT_BYTES:
        raise HTTPException(status_code=413, detail=f"转录 JSON 超过上限 {MAX_TRANSCRIPT_MB} MB")

    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="JSON 不是 UTF-8 编码") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"JSON 解析失败：{exc}") from exc

    try:
        normalized = summarizer.normalize_payload(payload)
    except summarizer.SummarizationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    task_id = uuid.uuid4().hex
    transcript_path = settings.transcript_path(task_id)
    transcript_path.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    filename = Path(file.filename or "transcript.json").name
    _register(
        TaskRecord(
            task_id=task_id,
            filename=filename,
            source_path=transcript_path,
            size_bytes=len(raw),
            status=TaskStatus.SUMMARIZING,
            progress=STAGE_RANGES[TaskStatus.SUMMARIZING][0],
            detail="已导入转录 JSON，排队总结",
            started_at=datetime.now(timezone.utc),
        )
    )
    logger.info("导入转录 JSON %s（%.1f KB）-> 任务 %s", filename, len(raw) / 1024, task_id)
    background_tasks.add_task(run_summarize, task_id)
    return SummarizeResponse(
        task_id=task_id,
        status=TaskStatus.SUMMARIZING,
        transcript_path=str(transcript_path),
        status_url=f"/status/{task_id}",
        result_url=f"/result/{task_id}",
        note="已导入转录 JSON 并开始总结，会调用 DeepSeek API 并产生费用",
    )


@app.get("/healthz", summary="自检")
def healthz() -> dict[str, Any]:
    with _tasks_lock:
        total = len(_tasks)
        pending_summary = sum(1 for r in _tasks.values() if r.status == TaskStatus.TRANSCRIBED)
    return {
        "status": "ok",
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "ffprobe": bool(shutil.which("ffprobe")),
        "media_io": {
            # 说话人分离走 waveform 解码，soundfile 可用即可；torchcodec 只有 path 模式才需要
            "soundfile": _module_available("soundfile"),
            "torchcodec": _module_available("torchcodec"),
        },
        "diarization": transcribe.diarization_status(),
        "asr_offline": transcribe.ASR_OFFLINE,
        "config": settings.as_dict(),
        "tasks_in_memory": total,
        "pending_summary": pending_summary,
    }
