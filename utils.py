from dataclasses import dataclass
import time
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class SilenceTrimConfig:
    enabled: bool = True
    threshold: float = 0.003
    padding_ms: int = 80
    frame_ms: int = 20
    hop_ms: int = 10


def trim_silence(
    wav: np.ndarray,
    sample_rate: int,
    config: SilenceTrimConfig = SilenceTrimConfig(),
) -> Tuple[np.ndarray, Dict[str, float]]:
    started_at = time.perf_counter()
    info = {
        "trim_removed_start_seconds": 0.0,
        "trim_removed_end_seconds": 0.0,
        "trim_elapsed_seconds": 0.0,
    }

    if not config.enabled:
        info["trim_elapsed_seconds"] = time.perf_counter() - started_at
        return wav, info

    audio = np.asarray(wav)
    if audio.size == 0:
        info["trim_elapsed_seconds"] = time.perf_counter() - started_at
        return audio, info

    amplitude = np.max(np.abs(audio), axis=1) if audio.ndim > 1 else np.abs(audio)
    peak = float(np.max(amplitude)) if amplitude.size else 0.0
    if peak <= 0.0:
        info["trim_elapsed_seconds"] = time.perf_counter() - started_at
        return audio, info

    threshold = min(config.threshold, peak * 0.02)
    frame_size = max(1, int(sample_rate * config.frame_ms / 1000))
    hop_size = max(1, int(sample_rate * config.hop_ms / 1000))

    active_frames = []
    for start in range(0, len(amplitude), hop_size):
        end = min(len(amplitude), start + frame_size)
        if np.max(amplitude[start:end]) > threshold:
            active_frames.append((start, end))

    if not active_frames:
        info["trim_elapsed_seconds"] = time.perf_counter() - started_at
        return audio, info

    padding = max(0, int(sample_rate * config.padding_ms / 1000))
    start = max(0, active_frames[0][0] - padding)
    end = min(len(audio), active_frames[-1][1] + padding)
    info["trim_removed_start_seconds"] = start / sample_rate
    info["trim_removed_end_seconds"] = (len(audio) - end) / sample_rate
    info["trim_elapsed_seconds"] = time.perf_counter() - started_at

    if start == 0 and end == len(audio):
        return audio, info
    return audio[start:end], info
