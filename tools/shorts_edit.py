#!/usr/bin/env python3
"""쇼츠 자동 편집기 — STYLE_GUIDE.md 스타일(자청/매운인문학 계열 정보형 쇼츠).

두 단계로 쓴다.
  1) 분석: 음성 구간 검출(VAD) + 받아쓰기(Whisper) → plan.json 생성
     python3 tools/shorts_edit.py plan input.mp4 -o work/plan.json --title "한국 1%들이 몰래 쓰는|제미나이 사용법"
  2) 렌더: plan.json대로 컷/줌/자막/타이틀/팝업을 입혀 1080x1920 mp4 출력
     python3 tools/shorts_edit.py render work/plan.json -o out.mp4
  한 번에: python3 tools/shorts_edit.py auto input.mp4 -o out.mp4 --title "윗줄|아랫줄"

plan.json의 모든 시간(cuts/captions/popups/broll/images)은 '원본 영상' 기준 초 단위다.
렌더할 때 컷 편집 후 타임라인으로 자동 변환된다.

자막 텍스트 강조 문법:  *노랑*  ~초록~  ^빨강^
"""
import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTS = os.path.join(ROOT, "fonts")
MODELS = os.path.join(ROOT, "models")
ASSETS = os.path.join(ROOT, "assets")

W, H, FPS = 1080, 1920, 30

# ASS 색상은 &HAABBGGRR
WHITE = "&H00FFFFFF"
BLACK = "&H00000000"
YELLOW = "&H0000E6FF"   # #FFE600
GREEN = "&H003CF03C"    # #3CF03C (초록 텍스트)
GREEN_BOX = "&H002EF54C"  # #4CF52E (초록 박스)
RED = "&H001E23E8"      # #E8231E
DARK_BOX = "&H201A1A1A"

INLINE = {"*": YELLOW, "~": GREEN, "^": RED}

# 템플릿별 레이아웃 (1080x1920 기준 px)
TEMPLATES = {
    # 풀스크린 토킹헤드 + 상단 고정 2줄 타이틀
    "fullscreen": {"video": (0, 0, W, H), "title_y": 230, "caption_y": 1090},
    # 어두운 별 배경 + 하단 둥근 카드 안에 영상, 상단은 헤더 그래픽 영역
    "card": {"video": (100, 760, 880, 900), "title_y": 470, "caption_y": 1290},
}


# ---------------------------------------------------------------- 공통 유틸

def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-4000:])
        raise SystemExit(f"명령 실패: {' '.join(cmd[:6])} ...")
    return r


def probe(path):
    info = json.loads(sh(["ffprobe", "-v", "error", "-print_format", "json",
                          "-show_format", "-show_streams", path]).stdout)
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
    return float(info["format"]["duration"]), has_audio


def load_audio_16k(path):
    import numpy as np
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", "16000",
                        "-f", "f32le", "-"], capture_output=True)
    if r.returncode != 0:
        raise SystemExit(r.stderr.decode()[-2000:])
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


# ---------------------------------------------------------------- 1) 분석

def find_whisper():
    for name in ("turbo", "medium", "small", "base", "tiny"):
        d = os.path.join(MODELS, f"sherpa-onnx-whisper-{name}")
        if os.path.isdir(d):
            files = os.listdir(d)
            pick = lambda key: next(
                (os.path.join(d, f) for f in sorted(files, key=lambda f: "int8" not in f)
                 if key in f and f.endswith(".onnx")), None)
            tokens = next(os.path.join(d, f) for f in files if f.endswith("tokens.txt"))
            return pick("encoder"), pick("decoder"), tokens
    raise SystemExit("Whisper 모델이 없습니다. ./setup.sh 를 먼저 실행하세요.")


def speech_segments(samples, sr=16000):
    import sherpa_onnx
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = os.path.join(MODELS, "silero_vad.onnx")
    cfg.silero_vad.threshold = 0.45
    cfg.silero_vad.min_silence_duration = 0.18
    cfg.silero_vad.min_speech_duration = 0.12
    cfg.silero_vad.max_speech_duration = 20
    cfg.sample_rate = sr
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=len(samples) / sr + 10)
    win = cfg.silero_vad.window_size
    for i in range(0, len(samples), win):
        vad.accept_waveform(samples[i:i + win])
    vad.flush()
    segs = []
    while not vad.empty():
        seg = vad.front
        segs.append((seg.start / sr, (seg.start + len(seg.samples)) / sr))
        vad.pop()
    return segs


def transcribe(samples, segs, sr=16000):
    import sherpa_onnx
    enc, dec, tok = find_whisper()
    rec = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=enc, decoder=dec, tokens=tok, language="ko", task="transcribe",
        num_threads=os.cpu_count() or 4)
    out = []
    for s, e in segs:
        st = rec.create_stream()
        st.accept_waveform(sr, samples[int(s * sr):int(e * sr)])
        rec.decode_stream(st)
        out.append(st.result.text.strip())
    return out


def clean_text(t):
    t = re.sub(r"[.,…·]+", " ", t)          # 쇼츠 자막은 마침표/쉼표 생략
    return re.sub(r"\s+", " ", t).strip()


def chunk_text(text, max_chars=10):
    """한 화면에 2~4어절(공백 제외 10자 이하)씩 끊는다."""
    words, chunks, cur = text.split(), [], []
    for w in words:
        if cur and len("".join(cur + [w])) > max_chars:
            chunks.append(" ".join(cur))
            cur = []
        cur.append(w)
        if re.search(r"[?!]$", w):
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def auto_emphasis(chunk):
    """숫자·%·금액이 들어간 어절은 노랑 강조."""
    return " ".join(f"*{w}*" if re.search(r"\d", w) else w for w in chunk.split())


def make_plan(src, title, title2_style, template, pad=0.06, gap_merge=0.12):
    dur, has_audio = probe(src)
    if not has_audio:
        raise SystemExit("오디오 트랙이 없는 영상입니다.")
    samples = load_audio_16k(src)
    segs = speech_segments(samples)
    texts = transcribe(samples, segs)

    # 컷: 말하는 구간만 남기되 앞뒤로 살짝 여유를 주고, 아주 짧은 틈은 합친다
    cuts = []
    for s, e in segs:
        s, e = max(0.0, s - pad), min(dur, e + pad)
        if cuts and s - cuts[-1][1] < gap_merge:
            cuts[-1][1] = e
        else:
            cuts.append([round(s, 3), round(e, 3)])

    captions = []
    for (s, e), text in zip(segs, texts):
        text = clean_text(text)
        if not text:
            continue
        chunks = chunk_text(text)
        total = sum(len(c.replace(" ", "")) for c in chunks)
        t = s
        for c in chunks:
            d = (e - s) * len(c.replace(" ", "")) / total
            captions.append({"start": round(t, 3), "end": round(t + d, 3),
                             "text": auto_emphasis(c), "style": "normal"})
            t += d

    lines = [l for l in (title or "").split("|") if l.strip()]
    title_items = []
    for i, l in enumerate(lines):
        title_items.append({"text": l.strip(), "style": "white" if i == 0 else title2_style})

    return {
        "source": os.path.abspath(src),
        "template": template,
        "zoom": "alternate",
        "title": title_items,
        "cuts": cuts,
        "captions": captions,
        "popups": [],
        "broll": [],
        "images": [],
        "bgm": None,
        "bgm_volume": 0.12,
    }


# ---------------------------------------------------------------- 2) 렌더

class Timeline:
    """원본 시간 → 컷 편집 후 시간."""

    def __init__(self, cuts):
        self.segs, o = [], 0.0
        for s, e in cuts:
            self.segs.append((s, e, o))
            o += e - s
        self.total = o

    def map(self, t):
        for s, e, o in self.segs:
            if t < s:
                return o
            if t <= e:
                return o + (t - s)
        return self.total


def ass_time(t):
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def fit_size(text, size, max_w=960):
    """한 줄이 화면 폭을 넘으면 글자 크기를 줄인다 (대략적인 글자 폭 추정)."""
    w = sum(0.3 if ch == " " else 0.62 if ord(ch) < 128 else 0.95
            for ch in text if ch not in INLINE) * size
    return size if w <= max_w else int(size * max_w / w)


def ass_text(text, base_color):
    text = text.replace("{", "（").replace("}", "）").replace("\n", r"\N")
    out, open_mark = [], None
    for ch in text:
        if ch in INLINE and (open_mark is None or open_mark == ch):
            if open_mark is None:
                out.append(r"{\c" + INLINE[ch] + "}")
                open_mark = ch
            else:
                out.append(r"{\c" + base_color + "}")
                open_mark = None
        else:
            out.append(ch)
    return "".join(out)


# name, font, size, primary, outline colour, back colour, bold, italic, borderstyle, outline, shadow
STYLES = [
    ("Cap", "Pretendard ExtraBold", 92, WHITE, BLACK, "&H80000000", 0, 0, 1, 7, 2),
    ("CapRed", "Black Han Sans", 150, RED, BLACK, "&H90000000", 0, 0, 1, 6, 6),
    ("Label", "Pretendard ExtraBold", 64, YELLOW, BLACK, "&H80000000", 0, 1, 1, 5, 0),
    ("TitleW", "Pretendard Black", 118, WHITE, BLACK, "&H80000000", 0, 0, 1, 8, 5),
    ("TitleY", "Pretendard Black", 118, YELLOW, BLACK, "&H80000000", 0, 0, 1, 8, 5),
    ("TitleG", "Pretendard Black", 112, BLACK, GREEN_BOX, GREEN_BOX, 0, 0, 3, 10, 0),
    ("TagBox", "Pretendard ExtraBold", 78, GREEN, DARK_BOX, DARK_BOX, 0, 0, 3, 18, 0),
    ("PopRed", "Black Han Sans", 140, RED, WHITE, "&H60000000", 0, 0, 1, 7, 4),
    ("PopY", "Black Han Sans", 130, YELLOW, BLACK, "&H60000000", 0, 0, 1, 7, 4),
    ("PopW", "Black Han Sans", 130, WHITE, BLACK, "&H60000000", 0, 0, 1, 7, 4),
]

TITLE_STYLE = {"white": ("TitleW", WHITE), "yellow": ("TitleY", YELLOW),
               "greenbox": ("TitleG", BLACK), "tagbox": ("TagBox", GREEN)}
POP_STYLE = {"red": ("PopRed", RED), "yellow": ("PopY", YELLOW), "white": ("PopW", WHITE)}

POP_IN = r"\fscx70\fscy70\t(0,90,\fscx108\fscy108)\t(90,150,\fscx100\fscy100)"
CAP_IN = r"\fscx92\fscy92\t(0,70,\fscx100\fscy100)"


def build_ass(plan, tl, path):
    tpl = TEMPLATES[plan["template"]]
    head = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}",
        "WrapStyle: 2", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
    ]
    for n, f, sz, pc, oc, bc, b, it, bs, o, shd in STYLES:
        head.append(f"Style: {n},{f},{sz},{pc},{pc},{oc},{bc},{b},{it},0,0,100,100,-2,0,"
                    f"{bs},{o},{shd},5,40,40,0,1")
    head += ["", "[Events]",
             "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    ev = []

    def add(layer, a, b, style, text, extra=""):
        if b - a >= 0.04:
            ev.append(f"Dialogue: {layer},{ass_time(a)},{ass_time(b)},{style},,0,0,0,,"
                      f"{{{extra}}}{text}")

    # 타이틀 (기본: 영상 전체 고정)
    y = plan.get("title_y", tpl["title_y"])
    for item in plan.get("title", []):
        style, base = TITLE_STYLE[item.get("style", "white")]
        size = fit_size(item["text"], item.get("size", 112 if style == "TitleG" else 118))
        a = tl.map(item["start"]) if "start" in item else 0
        b = tl.map(item["end"]) if "end" in item else tl.total
        iy = item.get("y", y)
        add(5, a, b, style, ass_text(item["text"], base), rf"\an5\pos(540,{iy})\fs{size}")
        y = iy + size + 24

    # 말 자막
    cy = plan.get("caption_y", tpl["caption_y"])
    for c in plan.get("captions", []):
        a, b = tl.map(c["start"]), tl.map(c["end"])
        st = c.get("style", "normal")
        if st == "red":
            add(3, a, b, "CapRed", ass_text(c["text"], RED),
                rf"\an5\pos(540,{cy + 40})\fs{fit_size(c['text'], 150)}" + CAP_IN)
        else:
            base = YELLOW if st == "yellow" else WHITE
            add(3, a, b, "Cap", ass_text(c["text"], base),
                rf"\an5\pos(540,{cy})\c{base}\fs{fit_size(c['text'], 92)}" + CAP_IN)
        if c.get("label"):
            lc = GREEN if c.get("label_color") == "green" else YELLOW
            ly = cy - (110 if st == "red" else 92)
            add(4, a, b, "Label", ass_text(c["label"], lc), rf"\an5\pos(540,{ly})\c{lc}")

    # 팝업 강조 (??????, 절대 하지 마세요 등)
    for p in plan.get("popups", []):
        style, base = POP_STYLE[p.get("style", "red")]
        size = rf"\fs{fit_size(p['text'], p.get('size', 140 if style == 'PopRed' else 130))}"
        add(6, tl.map(p["start"]), tl.map(p["end"]), style, ass_text(p["text"], base),
            rf"\an5\pos(540,{p.get('y', 700)}){size}" + POP_IN)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(head + ev) + "\n")


def ensure_card_assets():
    """카드 템플릿용 배경(어두운 남색+별 점)과 둥근 마스크/테두리 PNG."""
    from PIL import Image, ImageChops, ImageDraw, ImageFilter
    import random
    os.makedirs(ASSETS, exist_ok=True)
    bg_p = os.path.join(ASSETS, "card_bg.png")
    x, y, cw, ch = TEMPLATES["card"]["video"]
    if not os.path.exists(bg_p):
        bg = Image.new("RGB", (W, H))
        d = ImageDraw.Draw(bg)
        for i in range(H):
            k = i / H
            d.line([(0, i), (W, i)], fill=(int(10 + 14 * k), int(18 + 20 * k), int(40 + 30 * k)))
        glow = Image.new("RGB", (W, H), (0, 0, 0))
        ImageDraw.Draw(glow).ellipse([700, 380, 1300, 900], fill=(70, 70, 40))
        bg = ImageChops.add(bg, glow.filter(ImageFilter.GaussianBlur(160)))
        rnd = random.Random(7)
        d = ImageDraw.Draw(bg)
        for _ in range(140):
            px, py, r = rnd.randrange(W), rnd.randrange(H), rnd.choice([2, 2, 3, 4, 6])
            v = rnd.randrange(90, 200)
            d.ellipse([px - r, py - r, px + r, py + r], fill=(v, v, min(255, v + 20)))
        bg.save(bg_p)
    mask_p = os.path.join(ASSETS, "card_mask.png")
    frame_p = os.path.join(ASSETS, "card_frame.png")
    if not os.path.exists(mask_p):
        m = Image.new("L", (cw, ch), 0)
        ImageDraw.Draw(m).rounded_rectangle([0, 0, cw - 1, ch - 1], radius=44, fill=255)
        m.save(mask_p)
        fr = Image.new("RGBA", (cw + 16, ch + 16), (0, 0, 0, 0))
        ImageDraw.Draw(fr).rounded_rectangle([0, 0, cw + 15, ch + 15], radius=52,
                                             outline=(245, 245, 245, 255), width=10)
        fr.save(frame_p)
    return bg_p, mask_p, frame_p


def render(plan, out, work):
    os.makedirs(work, exist_ok=True)
    src = plan["source"]
    tpl = TEMPLATES[plan["template"]]
    vx, vy, vw, vh = tpl["video"]
    # cuts 항목: [시작, 끝] 또는 [시작, 끝, {"crop_x": 0~1, "fit": true, "zoom": 1.0}]
    raw = plan["cuts"] or [[0, probe(src)[0]]]
    cuts = [(c[0], c[1]) for c in raw]
    opts = [c[2] if len(c) > 2 else {} for c in raw]
    tl = Timeline(cuts)
    ass_p = os.path.join(work, "subs.ass")
    build_ass(plan, tl, ass_p)

    inputs = ["-i", src]
    fc = []
    zoom = plan.get("zoom", "alternate")
    top = plan.get("video_top", 0) if plan["template"] == "fullscreen" else 0
    cy_ = plan.get("crop_y", 0.35)   # 줌으로 세로가 남을 때 자를 위치 (0=위 기준)
    for i, (s, e) in enumerate(cuts):
        o = opts[i]
        z = o.get("zoom", 1.12 if (zoom == "alternate" and i % 2 == 1) else 1.0)
        cx = o.get("crop_x", plan.get("crop_x", 0.5))   # 가로 영상에서 사람이 있는 위치
        cut = f"[0:v]trim=start={s}:end={e},setpts=PTS-STARTPTS"
        if o.get("fit"):
            # 가로 화면 통째로 보여주기: 흐린 배경 위에 가로폭 맞춤
            fw = int(vw * z) // 2 * 2
            fc.append(f"{cut},split[fb{i}][ff{i}]")
            fc.append(f"[fb{i}]scale={vw}:{vh}:force_original_aspect_ratio=increase,"
                      f"crop={vw}:{vh},boxblur=30:2,eq=brightness=-0.12[fbb{i}]")
            fc.append(f"[ff{i}]scale={fw}:-2,crop=min(iw\\,{vw}):ih:(iw-min(iw\\,{vw}))*{cx}[ffs{i}]")
            fy = f"{top + 30}" if top else f"(H-h)/2-{o.get('lift', 0)}"
            fc.append(f"[fbb{i}][ffs{i}]overlay=(W-w)/2:{fy},setsar=1,fps={FPS}[v{i}]")
        else:
            ah = vh - top   # video_top: 위쪽을 검은 띠로 비워 타이틀 자리를 만든다
            tw, th = int(vw * z) // 2 * 2, int(ah * z) // 2 * 2
            pad = f",pad={vw}:{vh}:0:{top}:black" if top else ""
            fc.append(f"{cut},scale={tw}:{th}:force_original_aspect_ratio=increase,"
                      f"crop={vw}:{ah}:max(0\\,min(iw-{vw}\\,iw*{cx}-{vw}/2)):(ih-{ah})*{cy_}"
                      f"{pad},setsar=1,fps={FPS}[v{i}]")
        fc.append(f"[0:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS,"
                  f"afade=t=in:d=0.01,afade=t=out:st={max(0, e - s - 0.015):.3f}:d=0.015[a{i}]")
    fc.append("".join(f"[v{i}][a{i}]" for i in range(len(cuts))) +
              f"concat=n={len(cuts)}:v=1:a=1[vc][ac]")
    cur, n_in = "vc", 1

    # B롤: 해당 구간 동안 영상 영역을 다른 클립으로 덮음 (소리는 원본 유지)
    for j, b in enumerate(plan.get("broll", [])):
        inputs += ["-i", b["file"]]
        a, e = tl.map(b["start"]), tl.map(b["end"])
        fc.append(f"[{n_in}:v]trim=duration={e - a:.3f},setpts=PTS-STARTPTS+{a:.3f}/TB,"
                  f"scale={vw}:{vh}:force_original_aspect_ratio=increase,crop={vw}:{vh},"
                  f"setsar=1,fps={FPS}[br{j}]")
        fc.append(f"[{cur}][br{j}]overlay=0:0:eof_action=pass:"
                  f"enable='between(t,{a:.3f},{e:.3f})'[vb{j}]")
        cur, n_in = f"vb{j}", n_in + 1

    if plan["template"] == "card":
        bg_p, mask_p, frame_p = ensure_card_assets()
        inputs += ["-loop", "1", "-i", bg_p, "-loop", "1", "-i", mask_p, "-loop", "1", "-i", frame_p]
        bi, mi, fi = n_in, n_in + 1, n_in + 2
        n_in += 3
        fc.append(f"[{cur}]format=rgba[vr];[{mi}:v]format=gray[mk];[vr][mk]alphamerge[vm]")
        fc.append(f"[{bi}:v][vm]overlay={vx}:{vy}:shortest=1[c1]")
        fc.append(f"[c1][{fi}:v]overlay={vx - 8}:{vy - 8}:shortest=1[c2]")
        cur = "c2"

    # 이미지 오버레이 (아이콘, 리스트 카드 등)
    for j, im in enumerate(plan.get("images", [])):
        inputs += ["-loop", "1", "-i", im["file"]]
        a, e = tl.map(im["start"]), tl.map(im["end"])
        fc.append(f"[{n_in}:v]scale={im.get('w', 400)}:-1[im{j}]")
        fc.append(f"[{cur}][im{j}]overlay={im.get('x', '(W-w)/2')}:{im.get('y', 200)}:"
                  f"shortest=1:enable='between(t,{a:.3f},{e:.3f})'[vi{j}]")
        cur, n_in = f"vi{j}", n_in + 1

    ass_esc = ass_p.replace("\\", "/").replace(":", r"\:")
    fonts_esc = FONTS.replace(":", r"\:")
    fc.append(f"[{cur}]ass=filename='{ass_esc}':fontsdir='{fonts_esc}',format=yuv420p[vout]")

    # 오디오: 라우드니스 -14 LUFS, BGM 있으면 낮게 깔기
    if plan.get("bgm"):
        inputs += ["-stream_loop", "-1", "-i", plan["bgm"]]
        # 말할 때는 BGM이 자동으로 작아지게(사이드체인 더킹), 끝에서 페이드아웃
        fo = max(0.0, tl.total - 1.5)
        fc.append(f"[ac]asplit[vo][vk];[{n_in}:a]aformat=channel_layouts=stereo,"
                  f"volume={plan.get('bgm_volume', 0.12)},atrim=duration={tl.total:.3f},"
                  f"afade=t=out:st={fo:.3f}:d=1.5[bg0];"
                  f"[vk]aformat=channel_layouts=stereo[key];"
                  f"[bg0][key]sidechaincompress=threshold=0.03:ratio=4:attack=15:release=350[bgd];"
                  f"[vo][bgd]amix=inputs=2:duration=first:normalize=0[am]")
        acur = "am"
    else:
        acur = "ac"
    fc.append(f"[{acur}]loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[aout]")

    fc_path = os.path.join(work, "filter.txt")
    with open(fc_path, "w", encoding="utf-8") as f:
        f.write(";\n".join(fc))
    cmd = ["ffmpeg", "-y", "-v", "error", "-stats", *inputs, "-filter_complex_script", fc_path,
           "-map", "[vout]", "-map", "[aout]", "-t", f"{tl.total:.3f}",
           "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-r", str(FPS),
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)
    print(f"완료: {out}  ({tl.total:.1f}초, 컷 {len(cuts)}개)")


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "auto"):
        p = sub.add_parser(name)
        p.add_argument("input")
        p.add_argument("-o", "--output", required=True)
        p.add_argument("--title", default="", help='"윗줄|아랫줄"')
        p.add_argument("--title2-style", default="greenbox", choices=list(TITLE_STYLE))
        p.add_argument("--template", default="fullscreen", choices=list(TEMPLATES))
        p.add_argument("--no-zoom", action="store_true")
    r = sub.add_parser("render")
    r.add_argument("plan")
    r.add_argument("-o", "--output", required=True)
    a = ap.parse_args()

    if a.cmd in ("plan", "auto"):
        plan = make_plan(a.input, a.title, a.title2_style, a.template)
        if a.no_zoom:
            plan["zoom"] = "none"
        if a.cmd == "plan":
            os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
            with open(a.output, "w", encoding="utf-8") as f:
                json.dump(plan, f, ensure_ascii=False, indent=2)
            print(f"plan 저장: {a.output}  (컷 {len(plan['cuts'])}, 자막 {len(plan['captions'])})")
            return
        render(plan, a.output, os.path.splitext(a.output)[0] + "_work")
    else:
        with open(a.plan, encoding="utf-8") as f:
            plan = json.load(f)
        render(plan, a.output, os.path.splitext(a.output)[0] + "_work")


if __name__ == "__main__":
    main()
