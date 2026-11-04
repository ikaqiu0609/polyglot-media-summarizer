"""集中式配置：所有环境变量只在这里读一次，其余模块通过 `settings` 取值。

设计说明：
    audio / transcribe / summarizer 三个服务模块里仍保留了各自的模块级常量
    （便于单独 `python -m` 跑 CLI），但它们的关键参数都支持函数入参覆盖，
    main.py 一律显式传入 `settings` 里的值，因此运行时以本文件为准。

环境变量来源（后者不覆盖前者）：
    1. 系统环境变量 —— 云端（cnb.cool）由平台直接注入
    2. 项目根目录的 `.env` —— 本地 Windows / macOS / Linux 开发用（python-dotenv）

跨平台约定：
    - 路径全部用 pathlib.Path，代码里不出现写死的 /workspace、/tmp 之类绝对路径
    - 配置里的**相对路径一律相对项目根目录**解析，而不是当前工作目录
      （Windows 下从别的盘符启动 uvicorn 时，CWD 经常不是项目根）
    - 支持 `~/data`、`%USERPROFILE%\\data`、`$HOME/data` 等写法
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# app/config.py -> 项目根目录（Windows 上同样是绝对路径，例如 D:\code\polyglot-media-summarizer）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

try:
    from dotenv import load_dotenv
except ImportError:  # 未安装 python-dotenv 时退化为「只读系统环境变量」，不影响云端运行
    load_dotenv = None  # type: ignore[assignment]
else:
    # encoding="utf-8"：Windows 默认按 cp936 读文件，.env 里写中文注释会 UnicodeDecodeError
    # override=False：系统环境变量优先，云端注入的密钥不会被本地 .env 意外覆盖
    # 文件不存在时 load_dotenv 静默返回 False，容器/云端场景无副作用
    load_dotenv(dotenv_path=ENV_FILE, encoding="utf-8", override=False)


# HuggingFace 的日志噪音开关（必须在 huggingface_hub 首次导入前设置）。
# 本文件位于所有业务模块的导入链最上游，所以放这里最可靠；.env 里显式写同名变量可覆盖。
#   - 关闭 Windows 上「不支持 symlink」的长篇 UserWarning
#   - 关闭模型下载的 tqdm 进度条（它会和 logging 输出交错成半行）
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def resolve_path(raw: str | Path, base: Path = PROJECT_ROOT) -> Path:
    """把配置里的路径解析成跨平台的绝对路径。

    :param raw:  原始值，可含 `~`、`%USERPROFILE%`、`$HOME`，也可为相对路径
    :param base: 相对路径的基准目录（默认项目根目录）
    """
    text = str(raw).strip()
    # Windows 的 ntpath 会展开 %VAR%，POSIX 的 posixpath 会展开 $VAR，两边都安全
    expanded = Path(os.path.expandvars(os.path.expanduser(text)))
    if not expanded.is_absolute():
        expanded = base / expanded
    # Windows 上盘符路径也能正常 resolve；strict=False 允许目标尚不存在
    return expanded.resolve()

DEFAULT_MEDIA_EXTENSIONS: tuple[str, ...] = (
    ".mp4", ".mkv", ".mov", ".webm", ".flv", ".avi", ".m4v", ".ts",
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".amr", ".wma",
)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on", "y"}


def _env_path(name: str, default: Path) -> Path:
    """读取路径型环境变量：相对路径按项目根目录解析（不依赖 CWD）。"""
    raw = _env(name)
    return resolve_path(raw) if raw else default


def _env_set(name: str, default: tuple[str, ...]) -> frozenset[str]:
    raw = _env(name)
    if not raw:
        return frozenset(default)
    items = {item.strip().lower() for item in raw.replace(";", ",").split(",") if item.strip()}
    normalized = {item if item.startswith(".") else f".{item}" for item in items}
    return frozenset(normalized | set(default))


@dataclass(frozen=True)
class Settings:
    """运行时配置快照（frozen：进程内不可变，避免热更新带来状态不一致）。"""

    project_root: Path
    data_root: Path
    upload_dir: Path
    audio_dir: Path
    output_dir: Path

    allowed_extensions: frozenset[str]
    max_upload_mb: int

    # 处理参数（传给各服务的函数入参，覆盖其模块级默认值）
    chunk_minutes: float
    asr_language: str
    enable_diarization: bool
    summary_block_chars: int
    # 转录完成后是否自动调用 DeepSeek。默认 False：转录免费，总结要花钱，交给用户决定
    auto_summarize: bool

    # 运行时行为
    max_concurrent_tasks: int
    task_retention_hours: float
    log_level: str

    # 只读诊断信息（不含密钥本体）
    asr_model: str
    asr_align: str
    asr_batch_size: int
    vad_method: str
    diarization_model: str
    deepseek_model: str
    hf_token_present: bool
    deepseek_key_present: bool

    # ---------------- 派生属性 ----------------
    @property
    def max_upload_bytes(self) -> int:
        return max(1, self.max_upload_mb) * 1024 * 1024

    def task_audio_dir(self, task_id: str) -> Path:
        return self.audio_dir / task_id

    def upload_path(self, task_id: str, suffix: str) -> Path:
        return self.upload_dir / f"{task_id}{suffix}"

    def transcript_path(self, task_id: str) -> Path:
        return self.output_dir / f"{task_id}.transcript.json"

    def summary_path(self, task_id: str) -> Path:
        return self.output_dir / f"{task_id}.md"

    # ---------------- 行为 ----------------
    def ensure_dirs(self) -> None:
        for path in (self.data_root, self.upload_dir, self.audio_dir, self.output_dir):
            path.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, Any]:
        """脱敏快照，用于启动日志与 /healthz。"""
        return {
            "data_root": str(self.data_root),
            "upload_dir": str(self.upload_dir),
            "audio_dir": str(self.audio_dir),
            "output_dir": str(self.output_dir),
            "allowed_extensions": sorted(self.allowed_extensions),
            "max_upload_mb": self.max_upload_mb,
            "chunk_minutes": self.chunk_minutes,
            "asr_language": self.asr_language,
            "enable_diarization": self.enable_diarization,
            "summary_block_chars": self.summary_block_chars,
            "auto_summarize": self.auto_summarize,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "task_retention_hours": self.task_retention_hours,
            "asr_model": self.asr_model,
            "asr_align": self.asr_align,
            "asr_batch_size": self.asr_batch_size,
            "vad_method": self.vad_method,
            "diarization_model": self.diarization_model,
            "deepseek_model": self.deepseek_model,
            "hf_token_present": self.hf_token_present,
            "deepseek_key_present": self.deepseek_key_present,
        }

    @classmethod
    def from_env(cls) -> "Settings":
        data_root = _env_path("DATA_ROOT", PROJECT_ROOT / "data")
        return cls(
            project_root=PROJECT_ROOT,
            data_root=data_root,
            upload_dir=_env_path("UPLOAD_DIR", data_root / "uploads"),
            audio_dir=_env_path("AUDIO_DIR", data_root / "audio"),
            output_dir=_env_path("OUTPUT_DIR", data_root / "outputs"),
            allowed_extensions=_env_set("ALLOWED_EXTENSIONS", DEFAULT_MEDIA_EXTENSIONS),
            max_upload_mb=_env_int("MAX_UPLOAD_MB", 2048),
            chunk_minutes=_env_float("CHUNK_MINUTES", 15.0),
            asr_language=(_env("ASR_LANGUAGE", "auto") or "auto"),
            enable_diarization=_env_bool("ENABLE_DIARIZATION", True),
            summary_block_chars=_env_int("SUMMARY_BLOCK_CHARS", 8000),
            auto_summarize=_env_bool("AUTO_SUMMARIZE", False),
            max_concurrent_tasks=max(1, _env_int("MAX_CONCURRENT_TASKS", 1)),
            task_retention_hours=_env_float("TASK_RETENTION_HOURS", 24.0),
            log_level=(_env("LOG_LEVEL", "INFO") or "INFO").upper(),
            asr_model=_env("ASR_MODEL", "large-v3") or "large-v3",
            asr_align=(_env("ASR_ALIGN", "auto") or "auto").strip().lower(),
            asr_batch_size=_env_int("ASR_BATCH_SIZE", 16),
            vad_method=(_env("VAD_METHOD", "silero") or "silero").strip().lower(),
            diarization_model=_env("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
            or "pyannote/speaker-diarization-3.1",
            deepseek_model=_env("DEEPSEEK_MODEL", "deepseek-chat") or "deepseek-chat",
            hf_token_present=bool(_env("HF_TOKEN") or _env("HUGGINGFACE_TOKEN") or _env("HF_HUB_TOKEN")),
            deepseek_key_present=bool(_env("DEEPSEEK_API_KEY")),
        )


settings = Settings.from_env()
