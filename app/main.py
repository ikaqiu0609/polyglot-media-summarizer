"""FastAPI 主应用：上传 -> 后台流水线（抽取/分离/转录/总结）-> 状态查询与结果下载。

接口：
    POST /upload              上传媒体文件，立即返回 task_id（后台处理）
    GET  /status/{task_id}    查询状态与进度百分比
    GET  /result/{task_id}    下载 Markdown 总结
    GET  /healthz             配置与依赖自检（不返回任何密钥）

并发模型：
    BackgroundTasks 跑在线程池里，函数是同步阻塞的（subprocess + GPU 推理）。
    由于显存独占，用 Semaphore(MAX_CONCURRENT_TASKS，默认 1) 串行化，
    排队的任务状态保持 queued，进度为 0。
"""

from __future__ import annotations

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

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
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
    SUMMARIZING = "summarizing"
    COMPLETED = "completed"
    FAILED = "failed"


STAGE_LABELS: dict[TaskStatus, str] = {
    TaskStatus.QUEUED: "排队中",
    TaskStatus.EXTRACTING: "抽取音频",
    TaskStatus.DIARIZING: "说话人分离",
    TaskStatus.TRANSCRIBING: "语音转录",
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


def _update(task_id: str, **changes: Any) -> None:
    with _tasks_lock:
        record = _tasks.get(task_id)
        if record is None:
            return
        # 终态不可回退（后台线程迟到的一次进度回调不应把 completed 改回 transcribing）
        if record.status in _TERMINAL_STATUSES and "status" in changes and changes["status"] not in _TERMINAL_STATUSES:
            logger.debug("任务 %s 已终态（%s），忽略状态回退", task_id, record.status.value)
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
            and record.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}
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


def run_pipeline(task_id: str) -> None:
    """后台任务主体（同步阻塞，由 BackgroundTasks 放到线程池执行）。"""
    record = _get(task_id)
    if record is None:
        logger.error("任务 %s 不存在，跳过", task_id)
        return

    acquired_here = _pipeline_semaphore.acquire(blocking=False)
    if not acquired_here:
        _update(task_id, detail="排队中：等待前一个任务释放 GPU")
        _pipeline_semaphore.acquire()
        acquired_here = True

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

        # ---- 2. 转录 + 说话人分离 ----
        transcribed = transcribe.transcribe_task(
            chunks,
            language=settings.asr_language,
            do_diarize=settings.enable_diarization,
            progress=_progress_bridge(task_id, TaskStatus.TRANSCRIBING),
        )
        transcript_path = settings.transcript_path(task_id)
        transcript_path.write_text(
            json.dumps(transcribed, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _update(task_id, progress=STAGE_RANGES[TaskStatus.TRANSCRIBING][1], detail="转录完成，开始生成总结")

        # ---- 3. 两阶段总结 ----
        _update(
            task_id,
            status=TaskStatus.SUMMARIZING,
            progress=STAGE_RANGES[TaskStatus.SUMMARIZING][0],
            detail=f"DeepSeek 分段摘要（{transcribed.get('num_segments', 0)} 段）",
        )
        summary = summarizer.summarize(
            transcribed,
            block_chars=settings.summary_block_chars,
            save_to=settings.summary_path(task_id),
        )

        # ---- 4. 收尾 ----
        _update(
            task_id,
            status=TaskStatus.COMPLETED,
            progress=100.0,
            detail="已完成",
            finished_at=datetime.now(timezone.utc),
            summary_path=settings.summary_path(task_id),
            result={
                "summary_path": str(settings.summary_path(task_id)),
                "transcript_path": str(transcript_path),
                "num_segments": transcribed.get("num_segments", 0),
                "num_blocks": summary["meta"].get("num_blocks", 0),
                "speakers": transcribed.get("speakers", []),
                "language": transcribed.get("language"),
                "audio_duration_sec": total_audio,
                "tokens": summary["usage"],
            },
        )
        logger.info("任务 %s 完成：%s", task_id, settings.summary_path(task_id))

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
        if acquired_here:
            _pipeline_semaphore.release()


def _fail(task_id: str, message: str) -> None:
    logger.error("任务 %s 失败：%s", task_id, message)
    _update(
        task_id,
        status=TaskStatus.FAILED,
        detail="处理失败",
        error=message[:2000],
        finished_at=datetime.now(timezone.utc),
    )


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


# ---------------------------------------------------------------------------
# 应用
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    settings.ensure_dirs()
    logger.info("启动配置：%s", json.dumps(settings.as_dict(), ensure_ascii=False))
    if not shutil.which("ffmpeg"):
        logger.error("环境中找不到 ffmpeg，音频抽取会失败")
    if settings.enable_diarization and not settings.hf_token_present:
        logger.warning("未检测到 HF_TOKEN，任务将降级为「无说话人标签」")
    if not settings.deepseek_key_present:
        logger.warning("未检测到 DEEPSEEK_API_KEY，总结阶段会失败")
    yield
    logger.info("服务关闭")


app = FastAPI(
    title="Polyglot Media Summarizer",
    description="本地 ASR + 说话人分离 + DeepSeek 结构化总结",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/upload", response_model=UploadResponse, status_code=202, summary="上传媒体并排队处理")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> UploadResponse:
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

    _register(TaskRecord(task_id=task_id, filename=filename, source_path=dest, size_bytes=size))
    logger.info("接收上传 %s（%.1f MB）-> 任务 %s", filename, size / 1e6, task_id)

    # BackgroundTasks 会在响应返回后执行，run_pipeline 是同步函数，自动进线程池
    background_tasks.add_task(run_pipeline, task_id)

    return UploadResponse(
        task_id=task_id,
        status=TaskStatus.QUEUED,
        filename=filename,
        size_bytes=size,
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


@app.get("/healthz", summary="自检")
def healthz() -> dict[str, Any]:
    return {
        "status": "ok",
        "ffmpeg": bool(shutil.which("ffmpeg")),
        "ffprobe": bool(shutil.which("ffprobe")),
        "config": settings.as_dict(),
        "tasks_in_memory": len(_tasks),
    }
