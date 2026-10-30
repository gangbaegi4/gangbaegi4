#!/usr/bin/env python3
"""영상 전체에서 얼굴을 따라가며 모자이크한다.

사용법: python3 tools/face_mosaic.py input.mov -o input_mosaic.mp4

얼굴이 이미 흐리거나 옆/아래를 보고 있어도 놓치지 않도록 두 가지를 같이 쓴다.
  1) 자세 추정(BlazePose ONNX 모델, models/mp/)으로 처음 머리 위치를 찾는다.
  2) ViT 추적기(models/vittrack.onnx)로 머리를 따라간다. 놓치면 자세 추정으로 다시 찾는다.
둘 다 놓친 프레임은 앞뒤에서 찾은 위치를 이어서(보간) 덮는다.
"""
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")


class Pose:
    """BlazePose(ONNX) 최소 래퍼. 사람 영역(정사각형 + 회전)을 직접 넘겨 33개 관절점을 얻는다."""

    def __init__(self, path):
        self.net = cv2.dnn.readNet(path)
        self.names = self.net.getUnconnectedOutLayersNames()

    def infer(self, frame, cx, cy, size, angle):
        # 프레임 → 256x256 입력으로 가는 아핀 변환 (사람이 똑바로 서도록 회전)
        m = cv2.getRotationMatrix2D((cx, cy), angle, 256.0 / size)
        m[0, 2] += 128 - cx
        m[1, 2] += 128 - cy
        crop = cv2.warpAffine(frame, m, (256, 256), borderValue=(0, 0, 0))
        blob = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)[None] / 255.0
        self.net.setInput(blob)
        out = dict(zip(self.names, self.net.forward(self.names)))
        conf = float(out["Identity_1"][0, 0])
        lm = out["Identity"][0].reshape(39, 5).copy()
        lm[:, 3:] = 1 / (1 + np.exp(-lm[:, 3:]))
        inv = cv2.invertAffineTransform(m)
        lm[:, :2] = lm[:, :2] @ inv[:, :2].T + inv[:, 2]
        return conf, lm


def body_roi(lm, w, h):
    """관절점으로 다음 프레임용 사람 영역(중심, 크기, 회전각)을 만든다."""
    pts = lm[:33][lm[:33, 4] > 0.5][:, :2]
    if len(pts) < 8:
        return None
    x1, y1 = pts.min(0)
    x2, y2 = pts.max(0)
    hip = (lm[23, :2] + lm[24, :2]) / 2
    sho = (lm[11, :2] + lm[12, :2]) / 2
    v = sho - hip
    angle = np.degrees(np.arctan2(v[0], -v[1]))   # 어깨가 엉덩이 바로 위면 0도
    size = max(x2 - x1, y2 - y1) * 1.35
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, max(size, h * 0.4), angle])


def head_box_from_pose(lm):
    """lm: 39x5 [x, y, z, visibility, presence]. 0~10번이 얼굴 점, 11·12번이 어깨."""
    face = lm[0:11]
    good = face[face[:, 4] > 0.5]
    if len(good) < 3:
        return None
    cx, cy = good[:, 0].mean(), good[:, 1].mean()
    shoulder = np.linalg.norm(lm[11, :2] - lm[12, :2])
    spread = np.ptp(good[:, :2], axis=0).max()
    size = max(spread * 2.0, shoulder * 0.62)
    return np.array([cx, cy - size * 0.08, size])


def pixelate(frame, cx, cy, size, block=12):
    h, w = frame.shape[:2]
    half = size / 2
    x1, y1 = int(max(0, cx - half)), int(max(0, cy - half * 1.2))
    x2, y2 = int(min(w, cx + half)), int(min(h, cy + half * 1.1))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return
    roi = frame[y1:y2, x1:x2]
    bw = max(1, (x2 - x1) // block)
    bh = max(1, (y2 - y1) // block)
    small = cv2.resize(roi, (bw, bh), interpolation=cv2.INTER_AREA)
    frame[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--scale", type=float, default=1.35, help="모자이크 크기 배율 (여유)")
    ap.add_argument("--track", help="직접 보정한 머리 위치(npy, 프레임별 [x, y, 크기])로 바로 모자이크")
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pose = Pose(os.path.join(MODELS, "mp", "pose_estimation_mediapipe_2023mar.onnx"))
    # 처음 사람 영역: 화면 높이만 한 정사각형을 가로로 훑어 가장 확신도 높은 곳
    default_roi = None
    dw = 960
    dh = int(h * dw / w)
    yunet = cv2.FaceDetectorYN.create(os.path.join(MODELS, "face_yunet.onnx"), "", (dw, dh),
                                      0.7, 0.3, 5)

    if a.track:
        render_mosaic(a, np.load(a.track), np.zeros(n, int), fps, w, h)
        return

    # 1단계: 머리 추적 (ViT 추적기). 놓치면 자세 추정으로 머리를 다시 찾아 재시작한다.
    tp = cv2.TrackerVit_Params()
    tp.net = os.path.join(MODELS, "vittrack.onnx")
    found = np.full((n, 3), np.nan)
    tracker, last, size0 = None, None, None
    stats = {"pose": 0, "face": 0}

    def pose_head(frame, near):
        tries = []
        if near is not None:
            tries.append(np.array([near[0], near[1] + near[2] * 1.5, h * 1.0, 0.0]))
        tries.append(default_roi)
        for t in tries:
            for ang in (0.0, 90.0, -90.0):
                t2 = t.copy()
                t2[3] = ang
                if ang:
                    t2[2] = max(t2[2], w * 0.6)
                conf, lm = pose.infer(frame, *t2)
                if conf > 0.6:
                    hb = head_box_from_pose(lm)
                    if hb is not None and (near is None or
                                           np.linalg.norm(hb[:2] - near[:2]) < near[2] * 2.5):
                        return hb
        return None

    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            n = i
            found = found[:n]
            break
        if default_roi is None:
            best = max(((pose.infer(frame, x, h / 2, h * 1.1, 0)[0], x)
                        for x in np.linspace(h / 2, w - h / 2, 7)))
            default_roi = np.array([best[1], h / 2, h * 1.1, 0.0])
        got = None
        if tracker is not None:
            ok2, bx = tracker.update(frame)
            score = tracker.getTrackingScore()
            x, y, bw, bh = bx
            c = np.array([x + bw / 2, y + bh / 2])
            # 점수가 낮거나 박스가 비정상적으로 커지면(몸 전체로 번짐) 놓친 것으로 본다
            if ok2 and score > 0.35 and max(bw, bh) < size0 * 1.6 and \
                    (last is None or np.linalg.norm(c - last[:2]) < size0 * 1.5):
                got = np.array([c[0], c[1], size0])
            else:
                tracker = None
        if got is None and (i % 3 == 0 or last is None):
            hb = pose_head(frame, last)
            if hb is not None:
                stats["pose"] += 1
                size0 = size0 or max(hb[2], h * 0.17)
                got = np.array([hb[0], hb[1], size0])
                tracker = cv2.TrackerVit_create(tp)
                tracker.init(frame, (int(hb[0] - size0 / 2), int(hb[1] - size0 / 2),
                                     int(size0), int(size0)))
        if got is not None:
            found[i] = got
            last = got
        if i % 300 == 0:
            print(f"분석 {i}/{n}", file=sys.stderr)

    # 2단계: 튀는 값 제거(중앙값) → 빈 프레임은 앞뒤 위치로 보간 → 크기는 넉넉하게
    known = ~np.isnan(found[:, 0])
    if not known.any():
        raise SystemExit("사람을 찾지 못했습니다.")
    idx = np.arange(n)
    track = np.empty_like(found)
    for k in range(3):
        track[:, k] = np.interp(idx, idx[known], found[known, k])
    win = int(fps * 0.15) | 1
    pad = np.pad(track, ((win // 2, win // 2), (0, 0)), mode="edge")
    med = np.stack([np.median(pad[j:j + win], axis=0) for j in range(n)])
    # 크기는 주변 1초 중 최댓값을 써서 움직일 때 삐져나오지 않게
    big = np.pad(med[:, 2], (int(fps // 2), int(fps // 2)), mode="edge")
    med[:, 2] = [big[j:j + int(fps)].max() for j in range(n)]
    gap = (~known).astype(int)
    print(f"찾음 {known.sum()}/{n}프레임 (자세 {stats['pose']}, 얼굴 {stats['face']}), "
          f"나머지는 보간", file=sys.stderr)

    np.save(os.path.splitext(a.output)[0] + "_track.npy", med)   # 검수·보정용 머리 위치
    render_mosaic(a, med, gap, fps, w, h)


def render_mosaic(a, med, gap, fps, w, h):
    n = len(med)
    cap = cv2.VideoCapture(a.input)
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
         "-r", f"{fps}", "-i", "-", "-i", a.input, "-map", "0:v", "-map", "1:a?",
         "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p",
         "-c:a", "copy", "-shortest", a.output], stdin=subprocess.PIPE)
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        # 검출이 없던 구간은 위치가 불확실하니 조금 더 크게 덮는다
        extra = 1.25 if gap[max(0, i - 3):i + 4].any() else 1.0
        pixelate(frame, med[i, 0], med[i, 1], med[i, 2] * a.scale * extra)
        enc.stdin.write(frame.tobytes())
    enc.stdin.close()
    enc.wait()
    print(f"완료: {a.output}")


if __name__ == "__main__":
    main()
