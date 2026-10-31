#!/usr/bin/env python3
"""쇼츠용 효과음을 직접 합성하고, 타임라인에 맞춰 효과음 트랙 하나로 만든다 (저작권 걱정 없음).

종류
  impact : 빨강 펀치라인 — 쿵 하는 저음 + 짧은 타격음
  whoosh : 화면 전환·타이틀 — 바람 가르는 소리
  pop    : 라벨·팝업 — 뿅
  ding   : 숫자·정답 느낌 — 띵
"""
import wave

import numpy as np

SR = 48000


def _env(n, attack, release):
    t = np.arange(n) / SR
    return np.minimum(1, t / max(attack, 1e-4)) * np.exp(-t / release)


def _lowpass(x, cutoff):
    # 1차 저역통과를 벡터로 근사: 이동평균 여러 번
    k = max(1, int(SR / cutoff / 2))
    kernel = np.ones(k) / k
    for _ in range(3):
        x = np.convolve(x, kernel, mode="same")
    return x


def impact(rng):
    n = int(0.7 * SR)
    t = np.arange(n) / SR
    freq = 45 + 110 * np.exp(-t * 18)
    boom = np.sin(2 * np.pi * np.cumsum(freq) / SR) * _env(n, 0.002, 0.28)
    hit = _lowpass(rng.standard_normal(n), 2500) * _env(n, 0.001, 0.05) * 3
    return boom * 0.9 + hit * 0.5


def whoosh(rng):
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    x = rng.standard_normal(n)
    lo = _lowpass(x, 600)
    hi = x - _lowpass(x, 3000)
    k = t / t[-1]
    shape = np.sin(np.pi * k) ** 2
    return (lo * (1 - k) + hi * k * 0.5) * shape * 2.2


def pop(rng):
    n = int(0.16 * SR)
    t = np.arange(n) / SR
    freq = 900 * np.exp(-t * 14) + 350
    return np.sin(2 * np.pi * np.cumsum(freq) / SR) * _env(n, 0.001, 0.05)


def ding(rng):
    n = int(0.9 * SR)
    t = np.arange(n) / SR
    tone = sum(a * np.sin(2 * np.pi * f * t) for f, a in ((1568, 1), (3136, 0.35), (4704, 0.15)))
    return tone * _env(n, 0.002, 0.25) * 0.6


SOUNDS = {"impact": impact, "whoosh": whoosh, "pop": pop, "ding": ding}
GAIN = {"impact": 0.9, "whoosh": 0.45, "pop": 0.5, "ding": 0.35}


def build_track(events, total, path, volume=1.0):
    """events: [(초, 종류)] → 스테레오 wav 하나로 렌더."""
    rng = np.random.default_rng(11)
    cache = {k: f(rng) for k, f in SOUNDS.items()}
    out = np.zeros(int((total + 1) * SR))
    for t, kind in events:
        s = cache[kind] / (np.max(np.abs(cache[kind])) + 1e-9) * GAIN[kind] * volume
        # whoosh는 소리 정점이 강조 순간에 오도록 조금 앞당긴다
        start = int(max(0, t - (0.22 if kind == "whoosh" else 0)) * SR)
        end = min(len(out), start + len(s))
        out[start:end] += s[:end - start]
    out = np.clip(out[:int(total * SR)], -1, 1)
    stereo = np.stack([out, out], axis=1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((stereo * 32767).astype(np.int16).tobytes())
