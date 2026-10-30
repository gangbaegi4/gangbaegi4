#!/usr/bin/env bash
# 쇼츠 편집기 의존성 설치: Python 패키지 + 음성 인식 모델 (GitHub 릴리스에서 받음)
set -euo pipefail
cd "$(dirname "$0")"

pip install -q sherpa-onnx numpy pillow

REL=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models
mkdir -p models
[ -f models/silero_vad.onnx ] || curl -sSfL -o models/silero_vad.onnx "$REL/silero_vad.onnx"
# 더 빠르게 하려면 WHISPER=small ./setup.sh
WHISPER=${WHISPER:-turbo}
if [ ! -d "models/sherpa-onnx-whisper-$WHISPER" ]; then
  echo "Whisper $WHISPER 모델 다운로드 중..."
  curl -sSfL "$REL/sherpa-onnx-whisper-$WHISPER.tar.bz2" | tar xj -C models
  rm -rf "models/sherpa-onnx-whisper-$WHISPER/test_wavs"
fi
echo "준비 완료"
