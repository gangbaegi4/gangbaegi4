#!/usr/bin/env python3
"""저작권 걱정 없는 배경음악을 직접 합성한다 (운동/정보 쇼츠용 경쾌한 비트).

사용법: python3 tools/make_bgm.py -o bgm.wav --seconds 45 --bpm 118
구성: 킥(4박) + 클랩(2·4박) + 하이햇(엇박) + 베이스 + 부드러운 패드 코드 (Am-F-C-G)
"""
import argparse
import wave

import numpy as np

SR = 44100


def env(n, attack=0.005, release=0.2):
    t = np.arange(n) / SR
    return np.minimum(1, t / attack) * np.exp(-t / release)


def lowpass(x, cutoff):
    a = np.exp(-2 * np.pi * cutoff / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc = (1 - a) * v + a * acc
        y[i] = acc
    return y


def kick(n):
    t = np.arange(n) / SR
    freq = 50 + 90 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(freq) / SR) * env(n, 0.001, 0.18)


def noise_hit(n, release, rng, cutoff_hp=True):
    x = rng.standard_normal(n)
    if cutoff_hp:
        x = x - lowpass(x, 6000)   # 고역만 남겨 하이햇처럼
    return x * env(n, 0.001, release)


def saw(freq, n, detune=0.0):
    t = np.arange(n) / SR
    out = np.zeros(n)
    for d in (-detune, 0, detune):
        ph = (t * freq * (1 + d)) % 1.0
        out += 2 * ph - 1
    return out / 3


def note(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


def make(seconds, bpm, seed=3):
    rng = np.random.default_rng(seed)
    beat = 60 / bpm
    n_beats = int(np.ceil(seconds / beat)) + 4
    total = int(n_beats * beat * SR)
    drums = np.zeros(total)
    bass = np.zeros(total)
    pad = np.zeros(total)

    chords = [(57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62)]  # Am F C G
    roots = [33, 29, 36, 31]
    kick_s, clap_s, hat_s = kick(int(0.4 * SR)), noise_hit(int(0.25 * SR), 0.06, rng, False), \
        noise_hit(int(0.08 * SR), 0.025, rng)
    clap_s = lowpass(clap_s, 3000) * 1.5

    def put(buf, sig, start, gain):
        s = int(start * SR)
        e = min(total, s + len(sig))
        if s < total:
            buf[s:e] += sig[:e - s] * gain

    for b in range(n_beats):
        t0 = b * beat
        bar = b // 4
        put(drums, kick_s, t0, 0.9)
        if b % 4 in (1, 3):
            put(drums, clap_s, t0, 0.35)
        put(drums, hat_s, t0 + beat / 2, 0.18)
        put(drums, hat_s, t0, 0.07)
        # 베이스: 8분음표
        r = note(roots[(bar // 1) % 4])
        for k in range(2):
            n = int(beat / 2 * SR * 0.9)
            put(bass, lowpass(saw(r, n, 0.002), 400) * env(n, 0.005, 0.15), t0 + k * beat / 2, 0.55)
        # 패드: 마디 시작마다 코드
        if b % 4 == 0:
            n = int(4 * beat * SR)
            chord = sum(saw(note(m), n, 0.004) for m in chords[bar % 4])
            a = np.minimum(1, np.arange(n) / (0.3 * SR)) * np.minimum(1, (n - np.arange(n)) / (0.3 * SR))
            put(pad, lowpass(chord, 1200) * a, t0, 0.12)

    # 패드에 킥 사이드체인 펌핑 느낌
    pump = np.ones(total)
    for b in range(n_beats):
        s = int(b * beat * SR)
        k = min(total - s, int(beat * SR))
        if k > 0:
            pump[s:s + k] = 0.45 + 0.55 * np.minimum(1, np.arange(k) / (beat * SR * 0.6))
    mix = drums + bass + pad * pump
    mix = mix[:int(seconds * SR)]
    fade = int(1.5 * SR)
    mix[-fade:] *= np.linspace(1, 0, fade)
    mix /= np.max(np.abs(mix)) + 1e-9
    left = mix
    right = np.roll(mix, int(0.012 * SR)) * 0.9 + mix * 0.1   # 살짝 넓게
    return np.stack([left, right], axis=1) * 0.85


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--bpm", type=float, default=118)
    a = ap.parse_args()
    x = make(a.seconds, a.bpm)
    with wave.open(a.output, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    print(f"BGM 저장: {a.output} ({a.seconds:.0f}초, {a.bpm:.0f} BPM)")
