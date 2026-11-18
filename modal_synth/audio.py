"""WAV export. Normalization only - never clipping."""

from __future__ import annotations

import io
import wave

import numpy as np

TARGET_PEAK = 0.89  # linear full-scale fraction after normalization


def to_wav_bytes(
    samples: np.ndarray, sample_rate_hz: int, target_peak: float = TARGET_PEAK
) -> tuple[bytes, float, float]:
    """Encode mono float samples as 16-bit PCM WAV.

    Returns (wav_bytes, peak_before, gain_applied). The signal is scaled so
    its peak lands on target_peak; if the peak already equals target_peak
    the gain is 1. Values are rounded, not clipped: the solver's stability
    guarantees |samples| stays within the reported peak, so nothing ever
    exceeds full scale after normalization.
    """
    samples = np.asarray(samples, dtype=np.float64)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    gain = target_peak / peak if peak > 0.0 else 1.0
    # No clipping: peak * gain == target_peak < 1 by construction.
    pcm = np.round(samples * gain * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate_hz))
        wf.writeframes(pcm.tobytes())
    return buf.getvalue(), peak, gain
