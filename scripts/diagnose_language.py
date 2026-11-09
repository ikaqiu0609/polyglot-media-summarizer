"""诊断片内语言细分：打印每个探测窗口的 detect_language 原始返回与解析结果。

不用跑完整流水线，1~2 分钟出结果。用法（项目根目录）：

    .venv\\Scripts\\python.exe scripts\\diagnose_language.py "音频.m4a" --start 0 --duration 120

重点看三行：
    1. `soundfile: ...` 是否 MISSING；
    2. `首次返回: type=... repr=...` —— murmurai 到底返回了什么；
    3. 窗口序列里 60s 之后是否出现 en。
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services import audio, transcribe  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="诊断片内语言细分")
    parser.add_argument("media", help="音频/视频文件（m4a/mp4/wav...）")
    parser.add_argument("--start", type=float, default=0.0, help="相对媒体的起始秒")
    parser.add_argument("--duration", type=float, default=120.0, help="探测时长（秒）")
    parser.add_argument("--window", type=float, default=transcribe.LANG_PROBE_SEC, help="窗口长度（秒）")
    args = parser.parse_args()

    print(f"soundfile: {getattr(transcribe.sf, '__version__', 'MISSING')}")
    print(f"ASR_LANG_SPAN: {transcribe.LANG_SPAN_ENABLED}")
    print(f"ASR_LANG_MIN_CONF: {transcribe.LANG_SPAN_MIN_CONF}")
    print(f"ASR_LANG_PROBE_SEC: {transcribe.LANG_PROBE_SEC}")

    media = Path(args.media)
    if not media.exists():
        print(f"文件不存在：{media}")
        return 2

    with tempfile.TemporaryDirectory(prefix="lang-diag-") as tmp:
        wav = audio.extract_audio(media, dst=Path(tmp) / "full.16k.wav")
        model = transcribe.load_asr_model()
        detector = getattr(model, "detect_language", None)
        print(f"detect_language: {'可用' if callable(detector) else '不可用'}")

        total = audio.probe_duration(wav)
        end = min(args.start + args.duration, total)
        cursor = args.start
        windows: list[transcribe.LanguageWindow] = []
        first = True

        while cursor < end - 0.5:
            length = min(args.window, end - cursor)
            waveform = transcribe._read_waveform_span(wav, cursor, length)
            if waveform is None:
                print(f"{cursor:8.1f}s  读取失败（soundfile 与 wave 兜底都不可用）")
                cursor += length
                continue

            if first and callable(detector):
                try:
                    raw = detector(audio=waveform)
                except TypeError:
                    raw = detector(waveform)
                print(f"首次返回: type={type(raw).__name__} repr={raw!r}")
                print(f"解析结果: {transcribe._parse_language_result(raw)}")
                first = False

            language, confidence = transcribe.probe_language(model, waveform)
            print(f"{cursor:8.1f}s  {str(language):5}  {confidence:.2f}")
            windows.append(
                transcribe.LanguageWindow(
                    round(cursor, 3),
                    round(cursor + length, 3),
                    language if confidence >= transcribe.LANG_SPAN_MIN_CONF else None,
                    confidence,
                )
            )
            cursor += length

        spans = transcribe.merge_language_windows(windows)
        print("\nmerge_language_windows:")
        for span in spans:
            print(
                f"  {span.start:8.1f} ~ {span.end:8.1f}s  {span.language or 'auto'}  conf={span.confidence:.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
