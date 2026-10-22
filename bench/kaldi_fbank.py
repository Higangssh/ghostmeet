"""Kaldi fbank in numpy with the defaults sherpa-onnx gives kaldi-native-fbank for speaker
embeddings: the reference the app's Rust fbank is compared with (docs/spk2-diarization-design.md).
Fed to ERes2NetV2 on onnxruntime (with the mean over frames subtracted) it gives sherpa's
embeddings to a cosine of 0.9999998 or better."""
import numpy as np

def mel(f): return 1127.0 * np.log(1.0 + f / 700.0)

def mel_banks(num_bins, n_fft, rate, low, high):
    nyq = rate / 2
    if high <= 0: high += nyq
    fft_bin_width = rate / n_fft
    ml, mh = mel(low), mel(high)
    delta = (mh - ml) / (num_bins + 1)
    n = n_fft // 2
    banks = np.zeros((num_bins, n + 1), np.float32)   # knf: bins 0..n-1 used (padded_window_size/2)
    f = np.arange(n) * fft_bin_width
    m = mel(f)
    for b in range(num_bins):
        left, center, right = ml + b * delta, ml + (b + 1) * delta, ml + (b + 2) * delta
        up = (m - left) / (center - left); down = (right - m) / (right - center)
        w = np.maximum(0, np.minimum(up, down))
        banks[b, :n] = w
    return banks

def frames(x, flen=400, shift=160, snip_edges=False):
    n = len(x)
    if snip_edges:
        m = 0 if n < flen else 1 + (n - flen) // shift
        starts = np.arange(m) * shift
    else:
        m = (n + shift // 2) // shift
        starts = np.arange(m) * shift + shift // 2 - flen // 2
    out = np.empty((m, flen), np.float64)
    for i, s in enumerate(starts):
        idx = np.arange(s, s + flen)
        # reflect at the edges as kaldi does
        idx = np.where(idx < 0, -idx - 1, idx)
        idx = np.where(idx >= n, 2 * n - idx - 1, idx)
        out[i] = x[idx]
    return out

def fbank(x, rate=16000, num_bins=80, low=20.0, high=-400.0, snip_edges=False, preemph=0.97, scale=1.0, dtype=np.float32):
    x = np.asarray(x, np.float64) * scale
    fr = frames(x, snip_edges=snip_edges).astype(dtype)
    fr = fr - fr.mean(axis=1, keepdims=True)                 # remove_dc_offset
    pre = fr.copy()
    pre[:, 1:] = fr[:, 1:] - preemph * fr[:, :-1]
    pre[:, 0] = fr[:, 0] - preemph * fr[:, 0]
    a = 2 * np.pi / (400 - 1)
    win = (0.5 - 0.5 * np.cos(a * np.arange(400))) ** 0.85  # povey
    w = (pre * win.astype(dtype))
    n_fft = 512
    spec = np.abs(np.fft.rfft(w, n_fft)) ** 2               # power
    banks = mel_banks(num_bins, n_fft, rate, low, high)
    e = spec.astype(np.float32) @ banks.T
    return np.log(np.maximum(e, np.finfo(np.float32).eps)).astype(np.float32)
