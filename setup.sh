#!/usr/bin/env bash
# 쇼츠 편집기 의존성 설치: Python 패키지 + 음성 인식 모델 (GitHub 릴리스에서 받음)
set -euo pipefail
cd "$(dirname "$0")"

pip install -q sherpa-onnx numpy pillow opencv-python-headless

REL=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models
mkdir -p models
[ -f models/silero_vad.onnx ] || curl -sSfL -o models/silero_vad.onnx "$REL/silero_vad.onnx"
ZOO=https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models
mkdir -p models/mp
[ -f models/vittrack.onnx ] || curl -sSfL -o models/vittrack.onnx "$ZOO/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx"
[ -f models/face_yunet.onnx ] || curl -sSfL -o models/face_yunet.onnx "$ZOO/face_detection_yunet/face_detection_yunet_2023mar.onnx"
[ -f models/mp/pose_estimation_mediapipe_2023mar.onnx ] || curl -sSfL -o models/mp/pose_estimation_mediapipe_2023mar.onnx "$ZOO/pose_estimation_mediapipe/pose_estimation_mediapipe_2023mar.onnx"
# 더 빠르게 하려면 WHISPER=small ./setup.sh
WHISPER=${WHISPER:-turbo}
if [ ! -d "models/sherpa-onnx-whisper-$WHISPER" ]; then
  echo "Whisper $WHISPER 모델 다운로드 중..."
  curl -sSfL "$REL/sherpa-onnx-whisper-$WHISPER.tar.bz2" | tar xj -C models
  rm -rf "models/sherpa-onnx-whisper-$WHISPER/test_wavs"
fi
echo "준비 완료"
