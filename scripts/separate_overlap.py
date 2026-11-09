"""重叠语音分离（在独立的 pyannote.audio 3.3.2 环境里运行）。

为什么要单独一个环境：
    pyannote/speech-separation-ami-1.0 只支持 pyannote.audio==3.3.2，
    而主服务跑在 4.x（diarization 用的是 4.x 的 pipeline）。两个大版本不能共存，
    所以主进程用 subprocess 调这个脚本，脚本在 .venv-sep 里跑。

用法（通常由 app/services/separation.py 调用，不需要手动跑）：
    .venv-sep\\Scripts\\python.exe scripts\\separate_overlap.py ^
        --wav chunk.wav --start 567.9 --end 609.2 ^
        --out-dir data/audio/task/sep ^
        --manifest data/audio/task/sep/manifest.json

输出：
    out-dir/track_00.wav ... track_0N.wav   （16k mono PCM16，每个说话人一条）
    manifest.json                          （轨与说话人、时间映射）

退出码：
    0  成功
    2  环境/模型不可用（缺包、未同意 gated 条款、token 无效）
    3  没有可用的分离轨（纯静音/模型没输出）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _fail(code: int, message: str) -> int:
    print(f"[separate_overlap] {message}", file=sys.stderr)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description="对一段音频做说话人分离")
    parser.add_argument("--wav", required=True, help="16k mono PCM16 wav")
    parser.add_argument("--start", type=float, required=True, help="片段起点（秒）")
    parser.add_argument("--end", type=float, required=True, help="片段终点（秒）")
    parser.add_argument("--out-dir", required=True, help="输出目录")
    parser.add_argument("--manifest", required=True, help="manifest.json 路径")
    parser.add_argument("--model", default="pyannote/speech-separation-ami-1.0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-rms", type=float, default=0.004, help="低于该能量的轨视为空轨")
    args = parser.parse_args()

    try:
        import numpy as np
        import soundfile as sf
        import torch
    except ImportError as exc:
        return _fail(2, f"缺少依赖（{exc}）。请先运行 scripts/setup-separation-env.bat")

    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        return _fail(2, f"缺少 pyannote.audio（{exc}）。请先运行 scripts/setup-separation-env.bat")

    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
        or os.getenv("HF_HUB_TOKEN")
    )
    if not token:
        return _fail(2, "缺少 HF_TOKEN：分离模型是 gated，需要先在模型页同意条款并配置 token")

    wav = Path(args.wav)
    if not wav.exists():
        return _fail(2, f"音频不存在：{wav}")

    start = max(0.0, float(args.start))
    end = float(args.end)
    if end <= start:
        return _fail(2, f"时间区间非法：{start} ~ {end}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 16000
    try:
        data, rate = sf.read(
            str(wav),
            start=int(round(start * sample_rate)),
            frames=max(1, int(round((end - start) * sample_rate))),
            dtype="float32",
            always_2d=False,
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(2, f"读取音频失败：{exc}")

    if rate != sample_rate or data is None or len(data) == 0:
        return _fail(2, f"音频格式不支持（rate={rate}, samples={0 if data is None else len(data)}）")
    if data.ndim > 1:
        data = data.mean(axis=1)

    try:
        pipeline = Pipeline.from_pretrained(args.model, use_auth_token=token)
    except TypeError:
        # 少数 3.3.x 打包版本用 token=
        pipeline = Pipeline.from_pretrained(args.model, token=token)
    except Exception as exc:  # noqa: BLE001
        return _fail(
            2,
            f"加载分离模型失败：{exc}。常见原因：未在 {args.model} 页面同意条款、"
            "HF_TOKEN 无权限、或网络不可达",
        )

    if pipeline is None:
        return _fail(2, "分离模型返回空 pipeline（通常是 token 无权限或未同意 gated 条款）")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    try:
        pipeline.to(torch.device(device))
    except Exception as exc:  # noqa: BLE001
        print(f"[separate_overlap] 迁移到 {device} 失败（{exc}），改用 CPU", file=sys.stderr)
        device = "cpu"
        pipeline.to(torch.device("cpu"))

    waveform = torch.from_numpy(np.ascontiguousarray(data)).unsqueeze(0)
    payload = {"waveform": waveform, "sample_rate": sample_rate}

    try:
        with torch.inference_mode():
            output = pipeline(payload)
    except Exception as exc:  # noqa: BLE001
        return _fail(2, f"分离推理失败：{exc}")

    if isinstance(output, tuple) and len(output) == 2:
        diarization, sources = output
    else:
        return _fail(2, f"无法解析分离输出类型：{type(output)!r}")

    labels = list(diarization.labels()) if hasattr(diarization, "labels") else []
    source_data = getattr(sources, "data", None)
    if source_data is None:
        return _fail(2, "分离结果里没有 sources.data")

    source_data = np.asarray(source_data)
    if source_data.ndim == 1:
        source_data = source_data[:, None]

    # sources.data 通常是 (samples, speakers)；极少数版本是 (speakers, samples)
    n_samples = waveform.shape[1]
    if source_data.shape[0] != n_samples and source_data.shape[1] == n_samples:
        source_data = source_data.T

    tracks: list[dict] = []
    for index in range(source_data.shape[1]):
        track = np.asarray(source_data[:, index], dtype="float32")
        if track.size == 0:
            continue
        if track.size != n_samples:
            # 帧级输出：线性重采样到原长度，保证时间轴可对齐
            positions = np.linspace(0.0, 1.0, num=track.size, endpoint=False)
            target = np.linspace(0.0, 1.0, num=n_samples, endpoint=False)
            track = np.interp(target, positions, track).astype("float32")
        rms = float(np.sqrt(np.mean(np.square(track))) if track.size else 0.0)
        if rms < args.min_rms:
            continue
        speaker = labels[index] if index < len(labels) else f"SPEAKER_{index:02d}"
        filename = f"track_{len(tracks):02d}.wav"
        sf.write(
            str(out_dir / filename),
            np.clip(track, -1.0, 1.0),
            sample_rate,
            subtype="PCM_16",
        )
        tracks.append(
            {
                "index": len(tracks),
                "source_index": index,
                "speaker": str(speaker),
                "file": filename,
                "rms": round(rms, 5),
            }
        )

    if not tracks:
        return _fail(3, "没有可用的分离轨（可能整段是纯音乐/静音）")

    turns = []
    if hasattr(diarization, "itertracks"):
        turns = [
            {"start": round(float(turn.start), 3), "end": round(float(turn.end), 3), "speaker": str(speaker)}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

    manifest = {
        "model": args.model,
        "device": device,
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(end - start, 3),
        "tracks": tracks,
        "turns": turns,
    }
    Path(args.manifest).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[separate_overlap] 完成：{len(tracks)} 轨（device={device}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
