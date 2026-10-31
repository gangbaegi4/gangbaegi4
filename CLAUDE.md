# 쇼츠 자동 편집 작업 규칙

사용자가 영상을 보내면 `STYLE_GUIDE.md` 스타일로 바로 편집해서 결과 mp4를 돌려준다.

## 준비 (세션마다 처음 1번)
`./setup.sh`를 실행한다. 아래 두 가지를 설치한다.
- Whisper turbo 모델(~1GB)과 VAD 모델 → `models/`
- Python 패키지

## 편집 절차
1. **분석**
   ```
   python3 tools/shorts_edit.py plan <영상> -o work/<이름>/plan.json --title "윗줄|아랫줄"
   ```
   음성 구간으로 컷을 나누고 받아쓰기한 자막 초안을 만든다.
2. **plan.json 다듬기** (Claude가 직접 판단해서 고친다)
   - 받아쓰기 오타를 고치고, 한 줄 10자 이하로 다시 끊는다.
   - 키워드 한 단어는 `*노랑*`으로 강조한다. 숫자·금액은 자동으로 노랑 처리된다.
   - 주제가 바뀌는 지점에는 `label`을 단다. 예: `"[한국 출판 현실]"`. 질문이면 `label_color: "green"`.
   - 핵심 펀치라인은 `"style": "red"`로 지정한다.
   - 반전이나 리액션 지점에는 `popups`를 넣는다. 예: `"??????????"`
   - 사용자가 타이틀을 안 주면 내용을 보고 직접 정한다.
     - 1줄: 대상/상황
     - 2줄: 단정적인 충격 문구 (초록 박스 또는 노랑)
   - 실수, NG, 같은 말 반복 구간은 `cuts`에서 뺀다.
3. **렌더**
   ```
   python3 tools/shorts_edit.py render work/<이름>/plan.json -o out/<이름>.mp4
   ```
4. **검수**: 중간 프레임 몇 장을 캡처해서 자막 위치와 잘림을 확인한 다음 사용자에게 보낸다.

## 옵션
- `--template card`: 어두운 별 배경 + 둥근 카드 레이아웃
- `--title2-style yellow|greenbox|tagbox`: 타이틀 2번째 줄 스타일
- plan.json의 다른 항목:
  - `broll`: `[{start, end, file}]`
  - `images`: `[{start, end, file, x, y, w}]`
  - `bgm`, `bgm_volume`
- 레퍼런스 영상 수치 측정: `python3 tools/analyze_style.py ref.mp4`

`work/`, `out/`, `models/`는 커밋하지 않는다 (.gitignore).

## 참고 예시
- `examples/video1_plan.json`: 첫 편집(푸시업 영상)에서 사용자가 만족한 최종 plan.
  - 가로 원본이다: `crop_x`로 사람 위치를 맞추고, `video_top: 330`으로 위쪽을 검은 타이틀 띠로 비웠다. (450은 사람이 너무 아래에 있다고 해서 올림). 띠와 영상의 경계는 `top_fade: 220` 그라데이션으로 이었다.
  - 말하는 장면은 확대(1.0배)했다. 사용자가 시작 부분이 너무 확대됐다고 해서 1.12배 추가 확대는 뺐다.
  - 시범(몸 전체가 보여야 하는 장면)만 `fit` 전체 화면으로 처리했다.
  - BGM은 `tools/make_bgm.py`로 합성할 수 있다 (`bgm_volume: 0.5`, 말할 때 자동으로 작아짐). 이 영상의 최종본에서는 사용자가 BGM을 빼 달라고 해서 `bgm: null`이다.
  - 목소리: 사용자가 키워 달라고 해서 `"loudness": -11`(릴스용, 기본 -14)과 `"sfx_volume": 0.75`를 썼다. 렌더할 때 목소리에는 항상 컴프레서가 걸린다.
  - 효과음은 `"sfx": true`로 켠다 (`tools/make_sfx.py`로 합성). 빨강 자막은 쿵, 라벨·팝업은 뿅, 숫자는 띵, 화면이 바뀔 때는 휙.
- 영상 받는 방법: 사용자가 GitHub 릴리스(태그 `Video1`, `Video2`…)에 올린다. `get_release_by_tag`로 파일 주소를 확인하고 curl로 받는다.
- 얼굴 모자이크: `python3 tools/face_mosaic.py 원본 -o 원본_mosaic.mp4` → plan의 `source`를 결과 파일로 바꾼 다음 렌더한다.
  - 모델: `models/vittrack.onnx`, `models/face_yunet.onnx`, `models/mp/pose_estimation_mediapipe_2023mar.onnx` (opencv_zoo, media.githubusercontent.com에서 받음)
  - 푸시업처럼 엎드린 자세에서는 추적이 엇나갈 수 있다. 반드시 머리 위치를 프레임으로 확인한다.
  - 틀린 구간은 `_track.npy`를 고친 뒤 `--track`으로 다시 적용한다.
