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
