"""纯函数回归测试：切片规划、文本拼接/去重、说话人清洗、总结分块与 prompt 选择。

不依赖 GPU / 模型 / 网络，`python tests/test_pipeline_utils.py` 或 `pytest tests` 都能跑。
"""

from __future__ import annotations

import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # 允许直接 python tests/test_pipeline_utils.py
    sys.path.insert(0, str(ROOT))

from app.services import audio, summarizer, transcribe  # noqa: E402


# ---------------------------------------------------------------------------
# audio: 切片规划
# ---------------------------------------------------------------------------
def test_plan_chunks_plain() -> None:
    plans = audio.plan_chunks(1613.3, 600, silences=None, snap_window=0)
    assert [(p[0], p[1]) for p in plans] == [(0, 0.0), (1, 600.0), (2, 1200.0)]
    assert abs(plans[-1][1] + plans[-1][2] - 1613.3) < 1e-6


def test_plan_chunks_snaps_to_silence() -> None:
    plans = audio.plan_chunks(100, 50, silences=[(48.0, 54.0)], snap_window=20)
    assert plans[1][1] == 51.0  # 静音中点
    assert abs(plans[0][2] - 51.0) < 1e-6
    assert abs(plans[1][2] - 49.0) < 1e-6


def test_plan_chunks_ignores_far_or_out_of_window() -> None:
    # 静音离目标 25s（超出 window=20）→ 保持严格等分
    plans = audio.plan_chunks(100, 50, silences=[(23.0, 27.0)], snap_window=20)
    assert plans[1][1] == 50.0


def test_plan_chunks_stays_monotonic() -> None:
    # 第二个静音中点 35s 会把第二刀往前拉，但不能与前一片重叠
    plans = audio.plan_chunks(100, 40, silences=[(33.0, 37.0)], snap_window=20)
    starts = [p[1] for p in plans]
    assert starts == sorted(starts)
    assert all(p[2] > 0 for p in plans)


def test_plan_chunks_zero_duration() -> None:
    assert audio.plan_chunks(0, 600) == []


# ---------------------------------------------------------------------------
# audio: waveform 兜底解码
# ---------------------------------------------------------------------------
def test_load_waveform_stdlib_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x00" * 16000)

        waveform, sr = transcribe._load_waveform_stdlib(path, 16000)
        assert sr == 16000
        assert tuple(waveform.shape) == (1, 16000)


# ---------------------------------------------------------------------------
# transcribe: 文本拼接
# ---------------------------------------------------------------------------
def test_join_text_removes_cut_overlap() -> None:
    assert transcribe._join_text("the pro", "project news") == "the project news"


def test_join_text_cjk_without_space() -> None:
    assert transcribe._join_text("你好", "世界") == "你好世界"
    assert transcribe._join_text("hello", "world") == "hello world"


def test_join_text_identical_keeps_one() -> None:
    assert transcribe._join_text("same text", "same text") == "same text"


# ---------------------------------------------------------------------------
# transcribe: 说话人清洗 / 指派
# ---------------------------------------------------------------------------
def _turn(start: float, end: float, speaker: str) -> transcribe.DiarTurn:
    return transcribe.DiarTurn(start=start, end=end, speaker=speaker)


def test_merge_short_speakers_folds_ghost() -> None:
    turns = [_turn(0, 100, "A"), _turn(100, 160, "B"), _turn(160, 162, "GHOST")]
    merged = transcribe._merge_short_speakers(turns, total_duration=162)
    assert {t.speaker for t in merged} == {"A", "B"}
    assert merged[-1].speaker == "B"


def test_merge_short_speakers_keeps_single_speaker() -> None:
    turns = [_turn(0, 100, "A")]
    assert transcribe._merge_short_speakers(turns, 100) == turns


def test_nearest_speaker_respects_max_gap() -> None:
    original = transcribe.MAX_SPEAKER_ATTACH_GAP
    transcribe.MAX_SPEAKER_ATTACH_GAP = 2.0
    try:
        turns = [_turn(0, 10, "A")]
        assert transcribe._nearest_speaker(10.5, 11, turns) == "A"
        assert transcribe._nearest_speaker(40, 41, turns) is None
    finally:
        transcribe.MAX_SPEAKER_ATTACH_GAP = original


def test_fill_missing_labels() -> None:
    assert transcribe._fill_missing_labels([None, "A", None, "B", None]) == ["A", "A", "A", "B", "B"]


def test_split_segment_by_words() -> None:
    turns = [_turn(0, 2, "A"), _turn(2, 4, "B")]
    seg = {
        "start": 0.0,
        "end": 4.0,
        "text": "hello world",
        "words": [
            {"word": "hello", "start": 0.0, "end": 1.0},
            {"word": "world", "start": 2.5, "end": 3.5},
        ],
    }
    parts = transcribe._split_segment_by_words(seg, turns)
    assert parts is not None and len(parts) == 2
    assert [p["speaker"] for p in parts] == ["A", "B"]


def test_split_segment_single_speaker_returns_none() -> None:
    turns = [_turn(0, 10, "A")]
    seg = {
        "start": 0.0,
        "end": 1.0,
        "text": "hi there",
        "words": [
            {"word": "hi", "start": 0.0, "end": 0.3},
            {"word": "there", "start": 0.4, "end": 1.0},
        ],
    }
    assert transcribe._split_segment_by_words(seg, turns) is None


def test_speaker_stats_sorted_by_duration() -> None:
    turns = [_turn(0, 10, "A"), _turn(10, 15, "B"), _turn(15, 25, "A")]
    stats = transcribe._speaker_stats(turns)
    assert stats[0] == {"speaker": "A", "seconds": 20.0, "turns": 2}
    assert stats[1] == {"speaker": "B", "seconds": 5.0, "turns": 1}


# ---------------------------------------------------------------------------
# transcribe: 词级对齐（缓存命中路径）
# ---------------------------------------------------------------------------
class _FakeAlignModule:
    """只实现 align()，用来覆盖缓存命中分支（不需要 torch/HF）。"""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[tuple] = []
        self.fail = fail

    def align(self, segments, model, metadata, path, device=None):
        self.calls.append((model, metadata, path))
        if self.fail:
            raise RuntimeError("align boom")
        return [dict(seg, aligned=True) for seg in segments]


def _with_fake_mm(fake, func):
    original_mm = transcribe._mm
    transcribe._mm = lambda: fake
    transcribe._ALIGN_CACHE.clear()
    transcribe._ALIGN_UNAVAILABLE.clear()
    try:
        return func()
    finally:
        transcribe._mm = original_mm
        transcribe._ALIGN_CACHE.clear()
        transcribe._ALIGN_UNAVAILABLE.clear()


def test_align_segments_uses_cached_model() -> None:
    fake = _FakeAlignModule()

    def run():
        transcribe._ALIGN_CACHE["en"] = ("model", "meta")
        return transcribe._align_segments([{"text": "x"}], Path("a.wav"), "en")

    out = _with_fake_mm(fake, run)
    assert out[0]["aligned"] is True
    assert fake.calls and fake.calls[0][:2] == ("model", "meta")


class _FakeAlignModuleWithLoader(_FakeAlignModule):
    """模拟真实流程：第一次加载模型，第二次应命中缓存。"""

    def __init__(self) -> None:
        super().__init__()
        self.loads = 0

    def load_align_model(self, language_code, device=None):
        self.loads += 1
        return ("model", f"meta:{language_code}")


def test_align_segments_reuses_model_on_second_chunk() -> None:
    fake = _FakeAlignModuleWithLoader()

    def run():
        first = transcribe._align_segments([{"text": "a"}], Path("a.wav"), "en")
        second = transcribe._align_segments([{"text": "b"}], Path("b.wav"), "en")
        return first, second

    first, second = _with_fake_mm(fake, run)
    assert first[0]["aligned"] is True
    assert second[0]["aligned"] is True  # 修复前这里必然失败（mm 未定义）
    assert fake.loads == 1
    assert len(fake.calls) == 2


def test_align_failure_skips_language_afterwards() -> None:
    fake = _FakeAlignModule(fail=True)
    segments = [{"text": "x"}]

    def run():
        transcribe._ALIGN_CACHE["en"] = ("model", "meta")
        first = transcribe._align_segments(segments, Path("a.wav"), "en")
        second = transcribe._align_segments(segments, Path("a.wav"), "en")
        return first, second, "en" in transcribe._ALIGN_UNAVAILABLE

    first, second, unavailable = _with_fake_mm(fake, run)
    assert first == segments and second == segments
    assert unavailable is True
    assert len(fake.calls) == 1  # 第二次直接跳过，不再重复调用


# ---------------------------------------------------------------------------
# transcribe: 幻觉过滤 / 后处理
# ---------------------------------------------------------------------------
def test_filter_hallucinations() -> None:
    segments = [
        {"text": "repeated repeated repeated", "compression_ratio": 3.2},
        {"text": "thank you for watching", "no_speech_prob": 0.9, "avg_logprob": -1.5},
        {"text": "actual content here"},
        {"text": "actual content here"},
        {"text": "はい"},
        {"text": "はい"},
    ]
    kept = transcribe._filter_hallucinations(segments)
    assert [s["text"] for s in kept] == ["actual content here", "はい", "はい"]


def test_postprocess_merges_same_speaker_only() -> None:
    segments = [
        {"start": 0.0, "end": 1.0, "speaker": "Speaker 1", "text": "hello"},
        {"start": 1.1, "end": 2.0, "speaker": "Speaker 1", "text": "world"},
        {"start": 2.1, "end": 3.0, "speaker": "Speaker 2", "text": "other"},
    ]
    merged = transcribe.postprocess_segments(segments, gap_sec=0.8)
    assert len(merged) == 2
    assert merged[0]["text"] == "hello world"
    assert merged[1]["speaker"] == "Speaker 2"


# ---------------------------------------------------------------------------
# summarizer: 分块与三态 prompt 选择
# ---------------------------------------------------------------------------
def test_build_blocks_splits_on_segment_boundary() -> None:
    segments = [{"start": i, "end": i + 1, "speaker": "Speaker 1", "text": "x" * 300} for i in range(10)]
    blocks = summarizer.build_blocks(segments, max_chars=700)
    assert len(blocks) >= 4
    assert sum(len(b.segments) for b in blocks) == 10


def _payload(speaker: str, **extra) -> dict:
    base = {"segments": [{"start": 0, "end": 1, "speaker": speaker, "text": "hi"}], "duration_sec": 1}
    return dict(base, **extra)


def test_speaker_mode_three_states() -> None:
    labeled = _payload("Speaker 1", speakers=["Speaker 1"], diarized=True)
    assert summarizer._extract_segments(labeled)[1]["speaker_mode"] == "labeled"

    failed = _payload("UNKNOWN", speakers=[], diarized=False, diarization={"requested": True, "ok": False})
    assert summarizer._extract_segments(failed)[1]["speaker_mode"] == "multi_unlabeled"

    single = _payload("UNKNOWN", speakers=[], diarized=False, diarization={"requested": False, "ok": False})
    assert summarizer._extract_segments(single)[1]["speaker_mode"] == "single"

    imported = _payload("UNKNOWN", speakers=[])
    assert summarizer._extract_segments(imported)[1]["speaker_mode"] == "single"


def test_meta_line_matches_speaker_mode() -> None:
    multi = summarizer._postprocess_markdown(
        "# 标题", duration=65, speakers=[], language="multilingual", speaker_mode="multi_unlabeled"
    )
    assert "多人对话" in multi
    assert "00:01:05" in multi


if __name__ == "__main__":
    failures = 0
    for name, func in sorted(globals().items()):
        if not name.startswith("test_") or not callable(func):
            continue
        try:
            func()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
