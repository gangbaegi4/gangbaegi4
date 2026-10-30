#!/usr/bin/env python3
"""레퍼런스 영상의 편집 스타일 수치를 뽑는다 (ffmpeg/ffprobe만 사용).

사용법: python3 tools/analyze_style.py ref1.mp4 ref2.mp4 ... > style_report.json
측정 항목: 해상도/fps/길이, 컷 위치와 평균 컷 길이, 무음 구간(점프컷 여부), 음량(LUFS).
"""
import json
import re
import subprocess
import sys


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stderr


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
        capture_output=True, text=True,
    ).stdout
    info = json.loads(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = v["r_frame_rate"].split("/")
    return {
        "width": v["width"],
        "height": v["height"],
        "fps": round(int(num) / int(den), 2),
        "duration": round(float(info["format"]["duration"]), 2),
    }


def cuts(path, threshold=10.0):
    log = run(["ffmpeg", "-i", path, "-vf", f"scdet=threshold={threshold}", "-an", "-f", "null", "-"])
    return [round(float(t), 2) for t in re.findall(r"lavfi\.scd\.time:\s*([\d.]+)", log)]


def silences(path, noise="-35dB", min_dur=0.25):
    log = run(["ffmpeg", "-i", path, "-af", f"silencedetect=noise={noise}:d={min_dur}", "-vn", "-f", "null", "-"])
    starts = [float(x) for x in re.findall(r"silence_start:\s*([\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*([\d.]+)", log)]
    return [(round(s, 2), round(e, 2)) for s, e in zip(starts, ends)]


def loudness(path):
    log = run(["ffmpeg", "-i", path, "-af", "ebur128", "-vn", "-f", "null", "-"])
    m = re.search(r"Integrated loudness:\s*I:\s*(-?[\d.]+)", log)
    return float(m.group(1)) if m else None


def analyze(path):
    meta = probe(path)
    cut_times = cuts(path)
    bounds = [0.0] + cut_times + [meta["duration"]]
    shots = [round(b - a, 2) for a, b in zip(bounds, bounds[1:]) if b > a]
    sil = silences(path)
    return {
        "file": path,
        **meta,
        "cut_count": len(cut_times),
        "cuts": cut_times,
        "avg_shot_sec": round(sum(shots) / len(shots), 2) if shots else None,
        "cuts_per_10s": round(len(cut_times) / meta["duration"] * 10, 2),
        "silences": sil,
        "silence_total_sec": round(sum(e - s for s, e in sil), 2),
        "integrated_lufs": loudness(path),
    }


if __name__ == "__main__":
    print(json.dumps([analyze(p) for p in sys.argv[1:]], ensure_ascii=False, indent=2))
