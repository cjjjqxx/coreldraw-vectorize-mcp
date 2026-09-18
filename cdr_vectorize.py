"""Raster drawing -> clean line art + text labels, ready for CorelDRAW PowerTRACE.

Pipeline (verified on a 720x541 well-logging figure and a 600x450 blurry JPEG):
  1. prepare(): tone-stretch the source (faint strokes -> dark), Real-ESRGAN anime x4,
     downscale to 2x working scale; OCR the SOURCE (1x/2x/3x bicubic) for label
     candidates. Never OCR the SR image: SR hallucinates strokes on small blurry glyphs.
  2. finalize(): with human/LLM-confirmed labels, grow partial OCR boxes along their
     glyphs, wipe the labels while keeping underlines, crossing lines and leader lines
     that enter a box, and write sr2_no_text.png + labels.json for cdr_vectorize_com.py.

CLI:  python cdr_vectorize.py prepare  <image> <work_dir>
      python cdr_vectorize.py finalize <work_dir> <labels_confirmed.json>
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Real-ESRGAN (ncnn-vulkan build) lives in tools/realesrgan/ next to this file; override with CDR_TOOLS.
TOOLS = os.environ.get('CDR_TOOLS', os.path.join(HERE, 'tools'))
REALESRGAN = os.path.join(TOOLS, 'realesrgan', 'realesrgan-ncnn-vulkan.exe')
MODELS = os.path.join(TOOLS, 'realesrgan', 'models')
S = 2  # working scale relative to the source image


def imread(p, flag=cv2.IMREAD_GRAYSCALE):
    img = cv2.imdecode(np.fromfile(p, np.uint8), flag)
    if img is None:
        raise ValueError(f'cannot read image: {p}')
    return img


def imwrite(p, img):
    cv2.imencode(os.path.splitext(p)[1], img)[1].tofile(p)


# ---------------------------------------------------------------- stage 1

TONE_SPAN, TONE_GAMMA = 110, 1.0


def tone_stretch(gray, span=None, gamma=None):
    """Flatten uneven paper, then push faint strokes towards black before SR.
    Faint details (dashes, hatching at ~175-200 on a ~250 background) otherwise come out
    of SR as light squiggles that the black/white trace discards. Too strong a stretch
    thickens every line and fills small symbols, so span/gamma are a trade-off."""
    span = TONE_SPAN if span is None else span
    gamma = TONE_GAMMA if gamma is None else gamma
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31)))
    flat = np.clip(gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1) * 255, 0, 255)
    paper = float(np.percentile(flat, 60))
    white = min(paper - 14, 238)
    black = white - span
    out = np.clip((flat - black) / (white - black), 0, 1) ** gamma
    return (out * 255).astype(np.uint8)


def _sr2(inp, out_png, size, color=False):
    subprocess.run([REALESRGAN, '-i', inp, '-o', out_png, '-n', 'realesrgan-x4plus-anime', '-s', '4', '-m', MODELS],
                   check=True, capture_output=True, timeout=300)
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    return cv2.resize(imread(out_png, flag), size, interpolation=cv2.INTER_AREA)


def super_resolve(src_png, out_dir, span=None, gamma=None):
    """Two SR passes merged: the plain pass keeps true line weights and colors; the tone-stretched
    pass only contributes strokes the plain pass lost (faint dashes / hatching / thin grid lines).
    Colour is carried through; sr2.png (grey) and sr2_color.png (BGR) are both written."""
    src_c = imread(src_png, cv2.IMREAD_COLOR)
    h, w = src_c.shape[:2]
    size = (w * S, h * S)
    plain = _sr2(src_png, os.path.join(out_dir, 'sr4.png'), size, color=True)
    hsv = cv2.cvtColor(src_c, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = tone_stretch(hsv[:, :, 2], span, gamma)      # stretch lightness, keep hue/saturation
    enh = os.path.join(out_dir, 'src_enh.png')
    imwrite(enh, cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
    boosted = _sr2(enh, os.path.join(out_dir, 'sr4_enh.png'), size, color=True)
    g_plain = cv2.cvtColor(plain, cv2.COLOR_BGR2GRAY)
    g_boost = cv2.cvtColor(boosted, cv2.COLOR_BGR2GRAY)
    ink_b = (g_boost < 128).astype(np.uint8)
    src_up = cv2.resize(cv2.cvtColor(src_c, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_CUBIC)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink_b, 8)
    add = np.zeros_like(ink_b)
    for i in range(1, n):
        x, y, cw, ch, area = st[i]
        if area < 3 * S * S or max(cw, ch) > 40 * S:
            continue  # specks, or big line networks (those are fine in the plain pass)
        comp = lab[y:y + ch, x:x + cw] == i
        plain_ink = (g_plain[y:y + ch, x:x + cw] < 128) & comp
        frac = plain_ink.mean() / max(comp.mean(), 1e-9)
        pieces = cv2.connectedComponents(plain_ink.astype(np.uint8), connectivity=8)[0] - 1
        faint_in_source = src_up[y:y + ch, x:x + cw][comp].min() > 110   # crisp dots/lines are dark somewhere
        if faint_in_source and (frac < 0.4 or (frac < 0.8 and pieces >= 2)):
            add[y:y + ch, x:x + cw][comp] = 1
    add = cv2.dilate(add, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    merged = plain.copy()
    merged[add > 0] = boosted[add > 0]
    imwrite(os.path.join(out_dir, 'sr2_color.png'), merged)
    imwrite(os.path.join(out_dir, 'sr2.png'), cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY))
    return w, h


UNCERTAIN_MAX_SCORE = 0.9     # best score of the winning reading below this -> ask the caller to check
UNCERTAIN_MIN_VOTES = 4       # of up to 6 reads (2 image variants x 3 scales)
_DEGREE_TAIL = re.compile(r"^(\d{1,3})([0oO°º*96'\"]?)$")


def _norm_reading(t):
    return ''.join(t.replace('～', '~').replace('〜', '~').split())


def _fix_degree(gray, cand):
    """Map graticule labels: OCR reads the raised small circle of "37°" as 0/9/6/* or drops it
    ("370", "399", "29"). Measured on the glyphs: a small component in the upper half right of the
    digits is a degree sign; strip at most one misread trailing char and append "°"."""
    m = _DEGREE_TAIL.match(_norm_reading(cand['text']))
    if not m:
        return None
    x0, y0, x1, y1 = cand['box']
    crop = gray[max(y0 - 1, 0):y1 + 2, max(x0 - 1, 0):x1 + 2]
    if crop.size == 0 or crop.shape[0] < 5:
        return None
    thr, ink = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if thr > 225:
        return None
    n, _, st, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), 8)
    comps = [st[i] for i in range(1, n) if st[i][4] >= 2]
    if not comps:
        return None
    hmax = max(c[3] for c in comps)
    big = [c for c in comps if c[3] >= 0.6 * hmax]
    mid = min(c[1] + c[3] / 2.0 for c in big)
    right = max(c[0] + c[2] / 2.0 for c in big)
    small_top = [c for c in comps if c[3] <= 0.55 * hmax and c[1] + c[3] <= mid + 1 and c[0] + c[2] / 2.0 > right]
    if not small_top:
        return None
    digits, tail = m.group(1), m.group(2)
    note = None
    if not tail and len(big) < len(digits) and len(digits) >= 2:
        if len(big) >= 2:
            digits = digits[:-1]          # the circle itself was read as the last digit
        else:
            # glyphs merged at low resolution: the digit count cannot be measured (a width estimate
            # was tried and was wrong more often than right), keep all digits and ask the caller
            note = 'degree sign found but digits merged: check the number'
    fixed = digits + '°'
    if fixed != cand['text']:
        cand['readings'].append([fixed, 1.0, 'degree-fix'])
        cand['text'] = fixed
    return True, note


def ocr_candidates(src_bgr):
    """Multi-variant, multi-scale OCR on the source. Detections are clustered, the text is chosen by
    score-weighted vote, and every candidate carries a confidence so the caller knows what to check."""
    from rapidocr_onnxruntime import RapidOCR
    eng = RapidOCR(text_score=0.2)
    gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY) if src_bgr.ndim == 3 else src_bgr
    variants = [gray, cv2.createCLAHE(2.0, (8, 8)).apply(gray)]   # CLAHE lifts faint grey labels
    dets = []
    for v in variants:
        for f in (1, 2, 3):
            img = cv2.resize(v, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC) if f > 1 else v
            # no 180-degree angle classifier: drawing text is never upside down, and the classifier
            # flipped small labels ("90°" read as "006"/"606" and out-voted the correct read)
            res, _ = eng(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), use_cls=False, text_score=0.2, box_thresh=0.3)
            for quad, t, sc in res or []:
                xs = [p[0] / f for p in quad]; ys = [p[1] / f for p in quad]
                top = (quad[1][0] - quad[0][0], quad[1][1] - quad[0][1])   # top edge -> text baseline slant
                ang = math.degrees(math.atan2(top[1], top[0]))
                if abs(ang) < 4 or abs(ang) > 60:   # near-horizontal stays 0; absurd slants are noise
                    ang = 0.0
                dets.append({'text': t.strip(), 'score': float(sc), 'angle': round(ang, 1),
                             'box': [min(xs), min(ys), max(xs), max(ys)]})
    # vertical labels (chart axis titles): read the page rotated both ways and map boxes back.
    # Clockwise-rotated reads are bottom-to-top text (image angle -90), counter-clockwise top-to-bottom (+90).
    h0, w0 = gray.shape
    for rot, ang in ((cv2.ROTATE_90_CLOCKWISE, -90.0), (cv2.ROTATE_90_COUNTERCLOCKWISE, 90.0)):
        for f in (1, 2):
            r = cv2.rotate(gray, rot)
            img = cv2.resize(r, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC) if f > 1 else r
            res, _ = eng(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), use_cls=False, text_score=0.2, box_thresh=0.3)
            for quad, t, sc in res or []:
                pts = []
                for px, py in quad:
                    xr, yr = px / f, py / f
                    pts.append((yr, h0 - 1 - xr) if rot == cv2.ROTATE_90_CLOCKWISE else (w0 - 1 - yr, xr))
                xs = [p_[0] for p_ in pts]; ys = [p_[1] for p_ in pts]
                if (max(ys) - min(ys)) < 1.5 * (max(xs) - min(xs)) or len(t.strip()) < 2:
                    continue                                   # only genuinely vertical text lines
                dets.append({'text': t.strip(), 'score': float(sc), 'angle': ang,
                             'box': [min(xs), min(ys), max(xs), max(ys)]})

    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)

    clusters = []
    for dt in sorted(dets, key=lambda x: -x['score']):
        for c in clusters:
            if iou(c['box'], dt['box']) > 0.3:
                c['box'] = [min(c['box'][0], dt['box'][0]), min(c['box'][1], dt['box'][1]),
                            max(c['box'][2], dt['box'][2]), max(c['box'][3], dt['box'][3])]
                c['readings'].append([dt['text'], round(dt['score'], 2)])
                c['angles'].setdefault(_norm_reading(dt['text']), dt['angle'])
                break
        else:
            clusters.append({'box': list(dt['box']), 'angle': dt['angle'],
                             'angles': {_norm_reading(dt['text']): dt['angle']},
                             'readings': [[dt['text'], round(dt['score'], 2)]]})
    out = []
    for c in clusters:
        votes, best_form, best_score = {}, {}, {}
        for t, sc in c['readings']:
            k = _norm_reading(t)
            # score^4: many weak garbage reads must not out-vote one confident read ("006" beat "90°",
            # "量" beat "喜马拉雅" with a linear sum); measured best of max / linear / ^4 / ^8
            votes[k] = votes.get(k, 0.0) + sc ** 4
            if sc > best_score.get(k, -1):
                best_score[k], best_form[k] = sc, t
        win = max(votes, key=votes.get)
        n_win = sum(1 for t, _ in c['readings'] if _norm_reading(t) == win)
        cand = {'text': best_form[win], 'box': [int(v) for v in c['box']],
                'angle': c['angles'].get(win, c.get('angle', 0.0)),
                'readings': c['readings'], 'score': round(best_score[win], 2), 'votes': n_win,
                'agreement': round(votes[win] / sum(votes.values()), 2)}
        degree = _fix_degree(gray, cand)
        reasons = []
        score, n_agree = cand['score'], n_win
        if degree:
            # "37", "370", "37°" are the same reading of a degree label: count them together
            core = cand['text'][:-1]
            same = [r for r in c['readings'] if len(r) == 2 and _norm_reading(r[0]).startswith(core)
                    and len(_norm_reading(r[0])) <= len(core) + 1]
            n_agree = len(same)
            score = max((r[1] for r in same), default=0.0)
            if degree[1]:
                reasons.append(degree[1])
        if score < UNCERTAIN_MAX_SCORE:
            reasons.append(f'low OCR score {score}')
        if n_agree < UNCERTAIN_MIN_VOTES:
            reasons.append(f'only {n_agree}/{len([r for r in c["readings"] if len(r) == 2])} reads agree')
        cand['uncertain'] = bool(reasons)
        if reasons:
            cand['why'] = '; '.join(reasons)
        out.append(cand)
    return sorted(out, key=lambda c: (c['box'][1], c['box'][0]))


_ROMAN = re.compile(r'^[IVXLTAl1|\-·.]+$')
_CJK = re.compile(r'[\u4e00-\u9fff]')


def estimate_paper(bgr):
    """Paper colour: the dominant light colour (median of pixels near the lightness peak above 150)."""
    small = cv2.resize(bgr, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA) if bgr.shape[1] > 800 else bgr
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    light = g[g > 150]
    if light.size < 100:
        return np.array([255, 255, 255], np.float32)
    peak = np.argmax(np.bincount(light, minlength=256))
    sel = np.abs(g.astype(np.int16) - int(peak)) <= 8
    return np.median(small[sel], axis=0).astype(np.float32)


def normalize_paper(bgr):
    """Scans and photos of drawings have tinted paper (yellowish, grey). It was taken for content: a
    colour layer / grid over the whole page, and every 'is this paper' test failed. Scale colours so the
    paper becomes white. Global, not a local flat-field: a large filled box would be mistaken for
    background by a local estimate and wiped. Returns (image, paper_bgr or None if already white)."""
    paper = estimate_paper(bgr)
    if paper.min() >= 245 and paper.max() - paper.min() < 8:
        return bgr, None
    out = np.clip(bgr.astype(np.float32) * (255.0 / np.maximum(paper, 1.0)), 0, 255).astype(np.uint8)
    return out, [int(v) for v in paper]


def detect_mode(src_bgr):
    """'color' when the drawing has colour or filled regions, else 'lineart'. Chromatic share alone
    missed pastel / grey fills (flowchart boxes, table headers): line-art mode then dropped every fill.
    Filled = light-to-mid, non-paper pixels that form solid areas, not anti-aliased edges."""
    lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    content = src_bgr.min(axis=2) < 225
    if content.sum() < 100:
        return 'lineart'
    if float((chroma[content] > 25).mean()) > 0.05:
        return 'color'
    gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.boxFilter(gray, -1, (7, 7))
    std = np.sqrt(np.maximum(cv2.boxFilter(gray * gray, -1, (7, 7)) - mean * mean, 0))
    # flat only: blurred hatching / stipple is mid-grey too but striped (a blurred well log measured 5% "fill")
    fill = ((src_bgr.min(axis=2) < 240) & (gray >= 150) & (std < 5)).astype(np.uint8)
    solid = cv2.morphologyEx(fill, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return 'color' if float(solid.mean()) > 0.005 else 'lineart'


def suggested_labels(meta):
    """Label list for build from a prepare meta: the candidates select_labels() decided to use."""
    out = []
    for c in meta.get('candidates', []):
        if not c.get('use', True):
            continue
        if c.get('split_boxes') and '|' in c['text']:
            for part, box in zip(c['text'].split('|'), c['split_boxes']):
                out.append({'text': part, 'box': box, 'angle': 0.0})
        else:
            out.append({'text': c['text'], 'box': c['box'], 'angle': c.get('angle', 0.0)})
    return out


def select_labels(cands):
    """Decide per candidate whether it becomes a text object, so one call works without review.
    Roman-numeral-like tokens stay linework (OCR mangles them: IIIA, TIII, XI) and lone non-CJK
    characters are mostly symbol / hatching noise. Uncertain CJK is still used - and reported."""
    for c in cands:
        text = _norm_reading(c['text'])
        if not text:
            c['use'], c['skip_reason'] = False, 'empty'
        elif _ROMAN.match(text):
            c['use'], c['skip_reason'] = False, 'roman-numeral-like token: kept as traced linework'
        elif len(text) == 1 and not _CJK.match(text) and not text.isdigit():
            c['use'], c['skip_reason'] = False, 'single non-CJK character: treated as symbol/noise'
        elif (len(text) <= 3 and not _CJK.search(text) and c.get('uncertain')
              and not re.fullmatch(r'[<>]?[\d.,~\-]+(°|km|m|Ma)?', text)):
            # "0%", "OO", "1o": symbols (○ ⊙) read as characters. Not placed, listed for review.
            c['use'], c['skip_reason'] = False, 'single weak OCR read: short non-numeric token, likely a symbol'
        elif c.get('score', 1.0) < 0.5 and c.get('votes', 2) <= 1:
            # one weak read only: as often a symbol / colour patch / numeral read as a character as real
            # text; not placed, but listed in the review sheet so the caller can switch it on
            c['use'], c['skip_reason'] = False, 'single weak OCR read: not placed, check the review sheet'
        elif c.get('score', 1.0) < 0.5 and not _CJK.search(text):
            c['use'], c['skip_reason'] = False, 'low-confidence non-CJK reading'
        else:
            c['use'] = True
    return cands


def split_merged_label(src_gray, cand):
    """OCR reads neighbouring labels on one row as one line ("集流环测井仪器车测井仪器板"): one text object
    then sits wrong and overlaps its neighbours. For pure-CJK horizontal labels, cut the row at blank
    columns wider than 0.6 glyph height (inside a label CJK spacing is 0.1-0.3 h) and estimate each
    run's character count from its width (CJK pitch ~1.05 h). If the counts add up to the text the
    candidate becomes "a|b|c" (build splits the box); if not it is flagged for review."""
    text = _norm_reading(cand['text'])
    if len(text) < 4 or abs(cand.get('angle', 0)) > 4 or not all(_CJK.match(ch) for ch in text):
        return None
    x0, y0, x1, y1 = cand['box']
    crop = src_gray[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1]
    if crop.size == 0:
        return None
    paper = float(np.percentile(crop, 90))
    ink = (crop < paper - 60).astype(np.uint8)
    hh, ww = ink.shape
    ink &= 1 - cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(3, int(0.8 * hh))), np.uint8))  # underlines
    prof = ink.sum(axis=1)
    band = np.nonzero(prof >= 0.25 * max(prof.max(), 1))[0]          # text rows; leader lines are sparse
    if len(band) < 3:
        return None
    gh = float(band[-1] - band[0] + 1)
    cols = ink[band[0]:band[-1] + 1].any(axis=0)
    xs = np.nonzero(cols)[0]
    if len(xs) < 2:
        return None
    runs, start, gap = [], None, 0
    for i in range(xs[0], xs[-1] + 2):
        on = i <= xs[-1] and cols[i]
        if on:
            if start is None:
                start = i
            gap, end = 0, i
        elif start is not None:
            gap += 1
            if gap >= 0.35 * gh or i > xs[-1]:
                runs.append((start, end + 1)); start, gap = None, 0
    runs = [r for r in runs if r[1] - r[0] >= 0.45 * gh]              # leader-line slivers are not glyphs
    if len(runs) < 2:
        return None
    # the OCR text gives the total character count; share it by run width (largest remainder) and
    # accept only if every run then has a consistent character pitch (a fixed pitch from the glyph
    # height was off by one per run)
    widths = [b - a for a, b in runs]
    raw = [len(text) * w_ / float(sum(widths)) for w_ in widths]
    counts = [int(r) for r in raw]
    for i in sorted(range(len(raw)), key=lambda i: -(raw[i] - counts[i]))[:len(text) - sum(counts)]:
        counts[i] += 1
    if min(counts) < 2:
        return None      # a 1-character run is a letter-spaced title (喜 马拉雅) or a misread, not a label
    pitch = [w_ / c for w_, c in zip(widths, counts)]
    mean_pitch = sum(widths) / float(len(text))
    if max(abs(p_ - mean_pitch) for p_ in pitch) > 0.25 * mean_pitch:
        return f'several labels on one row? gaps do not fit {len(text)} characters'
    parts, k = [], 0
    for n in counts:
        parts.append(text[k:k + n]); k += n
    cand['readings'].append(['|'.join(parts), 1.0, 'row-split'])
    cand['text'] = '|'.join(parts)
    # keep the source-resolution cuts: re-splitting the merged box later on the SR image put the cuts
    # in the wrong gaps (测井仪器板 came out a third of its width, neighbours overlapped)
    cand['split_boxes'] = [[int(x0 + a), int(y0), int(x0 + b), int(y1)] for a, b in runs]
    return 'split'


def find_unlabeled_text(src_bgr, cands, max_items=8):
    """Glyph-like ink that no OCR candidate covers (faint grey single characters are often not even
    detected). Reported to the caller, never guessed: [{box, textness}] in source px."""
    gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    paper = float(np.percentile(gray, 70))
    ink = ((gray < paper - 45) & (chroma < 20)).astype(np.uint8)       # neutral dark: text/lines, not colour fills
    hs = sorted(c['box'][3] - c['box'][1] for c in cands if c.get('use') and abs(c.get('angle', 0)) < 4)
    H = float(np.median(hs)) if hs else 12.0
    covered = np.zeros_like(ink)
    for c in cands:
        x0, y0, x1, y1 = c['box']
        covered[max(y0 - 2, 0):y1 + 3, max(x0 - 2, 0):x1 + 3] = 1
    n, lbl, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    keep = np.zeros_like(ink)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if max(w, h) > 2.2 * H or a < 3:           # lines / frames / specks
            continue
        keep[lbl == i] = 1
    k = max(1, int(round(0.35 * H)))
    groups = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(groups, 8)
    found = []
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        if not (0.6 * H <= h <= 1.8 * H and 0.6 * H <= w <= 10 * H):
            continue
        box = [int(x), int(y), int(x + w), int(y + h)]
        if covered[y:y + h, x:x + w].mean() > 0.2:
            continue
        if (chroma[y:y + h, x:x + w] >= 25).mean() > 0.05:
            continue                                  # dark rims of colour patches
        sub = keep[y:y + h, x:x + w]
        fill = float(sub.mean())
        pn, plbl, pst, _ = cv2.connectedComponentsWithStats(sub, 8)
        if not (0.08 <= fill <= 0.55) or pn - 1 < 2:   # CJK glyphs are several strokes, symbols one ring
            continue
        # map symbols are rings (a hole inside a piece); roman numerals are mostly vertical bars.
        # (solidity was tried: meaningless on 10-15 px glyphs)
        cnts, hier = cv2.findContours(sub.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = sum(cv2.contourArea(cnts[k]) for k in range(len(cnts)) if hier is not None and hier[0][k][3] >= 0)
        if holes >= 0.08 * w * h:
            continue
        bars = sum(int(pst[j][4]) for j in range(1, pn) if pst[j][3] >= 2.2 * pst[j][2])
        if bars >= 0.5 * sub.sum():
            continue
        branched = float(sub.sum() - bars)
        found.append({'box': box, 'textness': round(branched / max(sub.sum(), 1), 2)})
    found.sort(key=lambda f: -f['textness'])
    return found[:max_items]


def review_sheet(src_bgr, cands, missing, path):
    """One image the calling AI can read to verify/correct every uncertain label (L#) and every piece
    of unrecognized text (M#): source crop enlarged, current reading, reason."""
    from PIL import Image, ImageDraw, ImageFont
    items = [(f'L{i}', c['box'], c['text'], c.get('why', '')) for i, c in enumerate(cands)
             if c.get('use') and c.get('uncertain')]
    items += [(f'L{i}', c['box'], c['text'], '未放置: ' + c.get('why', '')) for i, c in enumerate(cands)
              if not c.get('use') and c.get('skip_reason', '').startswith('single weak')]
    items += [(f'M{j}', m['box'], '?', 'glyph-like ink not covered by any label') for j, m in enumerate(missing)]
    if not items:
        return None
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 18)
    except OSError:
        font = ImageFont.load_default()
    H_img, W_img = src_bgr.shape[:2]
    rows = []
    for tag, (x0, y0, x1, y1), text, why in items:
        pad = max(4, int(0.4 * (y1 - y0)))
        cx0, cy0 = max(x0 - pad, 0), max(y0 - pad, 0)
        crop = src_bgr[cy0:min(y1 + pad, H_img), cx0:min(x1 + pad, W_img)]
        z = 64.0 / max(crop.shape[0], 1)
        crop = cv2.resize(crop, None, fx=z, fy=z, interpolation=cv2.INTER_CUBIC)
        # draw the box: the caller must see whether it covers the whole label ("松" boxed, "潘" outside)
        cv2.rectangle(crop, (int((x0 - cx0) * z), int((y0 - cy0) * z)),
                      (int((x1 - cx0) * z), int((y1 - cy0) * z)), (0, 0, 255), 1)
        crop = crop[:, :420]
        rows.append((tag, crop, text, why))
    row_h, width = 76, 900
    sheet = Image.new('RGB', (width, row_h * len(rows) + 8), 'white')
    draw = ImageDraw.Draw(sheet)
    for r, (tag, crop, text, why) in enumerate(rows):
        y = 4 + r * row_h
        draw.text((6, y + 24), tag, fill=(200, 0, 0), font=font)
        sheet.paste(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), (60, y + 4))
        draw.text((490, y + 10), f'读作: {text}', fill=(0, 0, 0), font=font)
        draw.text((490, y + 40), why[:36], fill=(110, 110, 110), font=font)
        draw.line([(0, y + row_h - 2), (width, y + row_h - 2)], fill=(220, 220, 220))
    sheet.save(path)
    return path


MAX_WORK_MPX = 6.0   # working image (source x S) cap. PowerTRACE returned 0 curves on a 20 Mpx mask and
                     # took 27 s per layer at 14 Mpx; drawings up to ~2 Mpx were fine and stay untouched.


def prepare(image_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    src_png = os.path.join(out_dir, 'src.png')
    img = imread(image_path, cv2.IMREAD_COLOR)                # colour kept; grey reads still work
    ih, iw = img.shape[:2]
    source_scale = 1.0
    work_mpx = iw * ih * S * S / 1e6
    if work_mpx > MAX_WORK_MPX:
        # large, usually crisp exports (report figures, 300 dpi scans): shrink before SR so every layer
        # stays traceable; all boxes and the page then refer to src.png
        source_scale = (MAX_WORK_MPX / work_mpx) ** 0.5
        img = cv2.resize(img, (max(1, int(iw * source_scale)), max(1, int(ih * source_scale))),
                         interpolation=cv2.INTER_AREA)
    img, paper_bgr = normalize_paper(img)
    imwrite(src_png, img)
    w, h = super_resolve(src_png, out_dir)
    src_bgr = imread(src_png, cv2.IMREAD_COLOR)
    cands = select_labels(ocr_candidates(src_bgr))
    src_gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    for c in cands:
        if c.get('use'):
            r = split_merged_label(src_gray, c)
            if r and r != 'split':
                c['uncertain'] = True
                c['why'] = '; '.join(x for x in (c.get('why'), r) if x)
    missing = find_unlabeled_text(src_bgr, cands)
    sheet = review_sheet(src_bgr, cands, missing, os.path.join(out_dir, 'label_review.png'))
    meta = {'source': image_path, 'src_size': [w, h], 'scale': S, 'candidates': cands,
            'unlabeled_text': missing, 'review_sheet': sheet, 'mode': detect_mode(src_bgr),
            'source_scale': round(source_scale, 4), 'input_size': [iw, ih],
            'paper_rgb': paper_bgr[::-1] if paper_bgr else None}
    with open(os.path.join(out_dir, 'ocr_candidates.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    vis = cv2.cvtColor(imread(src_png), cv2.COLOR_GRAY2BGR)
    vis = cv2.resize(vis, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    for i, c in enumerate(cands):
        x0, y0, x1, y1 = [v * 2 for v in c['box']]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(vis, str(i), (x0, max(y0 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    imwrite(os.path.join(out_dir, 'ocr_preview.png'), vis)
    return meta


# ---------------------------------------------------------------- stage 2

def _rule_mask(ink):
    """Long horizontal AND vertical rules. Label growth used only the horizontal mask, so a label next to
    a chart axis or table border grew along the vertical line ("0.6" came out twice its size)."""
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 18 * S)))
    return _line_mask(ink) | v


def _line_mask(ink):
    return cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (18 * S, 1)))


def _glyph_components(ink_nh, box, glyph_h):
    """Components that look like glyph pieces: short, and inside the text band."""
    x0, y0, x1, y1 = box
    H, W = ink_nh.shape
    X0, X1 = max(x0 - 6 * glyph_h, 0), min(x1 + 6 * glyph_h, W)
    Y0, Y1 = max(y0 - glyph_h, 0), min(y1 + glyph_h, H)
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink_nh[Y0:Y1, X0:X1], 8)
    mask = np.zeros_like(ink_nh)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        gy0, gy1 = y + Y0, y + Y0 + h
        if h <= 1.3 * glyph_h and w <= 1.6 * glyph_h and gy0 >= y0 - 0.35 * glyph_h and gy1 <= y1 + 0.35 * glyph_h:
            mask[Y0:Y1, X0:X1][lab == i] = 1
    return mask


def _glyph_band(ink_nh, box):
    """(glyph_h, band_y0, band_y1) from components lying fully inside the box. Lines that merely
    cross the box (leaders of neighbouring labels) would otherwise inflate the glyph height."""
    x0, y0, x1, y1 = box
    sub = ink_nh[y0:y1 + 1, x0:x1 + 1]
    n, _, st, _ = cv2.connectedComponentsWithStats(sub, 8)
    inner = [st[i] for i in range(1, n)
             if st[i][0] > 0 and st[i][1] > 0 and st[i][0] + st[i][2] < sub.shape[1] and st[i][1] + st[i][3] < sub.shape[0]
             and st[i][4] >= 4]
    if inner:
        top = min(s[1] for s in inner); bot = max(s[1] + s[3] for s in inner)
        return max(bot - top, 4), y0 + top, y0 + bot
    rows = np.nonzero(sub.any(axis=1))[0]
    if len(rows):
        return max(int(rows.max() - rows.min() + 1), 4), y0 + int(rows.min()), y0 + int(rows.max())
    return max(y1 - y0, 4), y0, y1


def expand_to_text(ink_nh, box, n_chars):
    """Grow a partial OCR box sideways, following glyph-like components only (not leader lines)."""
    x0, y0, x1, y1 = box
    glyph_h, gy0, gy1 = _glyph_band(ink_nh, box)
    target = n_chars * glyph_h * 1.05
    if x1 - x0 >= target * 0.85:
        return [x0, y0, x1, y1]
    glyphs = _glyph_components(ink_nh, (x0, gy0, x1, gy1), glyph_h)
    band = glyphs[gy0:gy1 + 1].any(axis=0)
    gap_max = int(0.8 * glyph_h)
    x, gap = x0 - 1, 0
    while x > 0 and gap <= gap_max and x1 - x < target * 1.1:
        if band[x]:
            x0, gap = x, 0
        else:
            gap += 1
        x -= 1
    x, gap = x1 + 1, 0
    while x < ink_nh.shape[1] - 1 and gap <= gap_max and x - x0 < target * 1.1:
        if band[x]:
            x1, gap = x, 0
        else:
            gap += 1
        x += 1
    return [x0, y0, x1, y1]


def _split_box(ink_nh, box, parts):
    """Split one OCR line box into len(parts) boxes at the real gaps between glyph runs,
    choosing for each expected (character-count proportional) cut the nearest wide gap."""
    x0, y0, x1, y1 = box
    glyph_h, gy0, gy1 = _glyph_band(ink_nh, box)
    g = _glyph_components(ink_nh, (x0, gy0, x1, gy1), glyph_h)
    cols = g[gy0:gy1 + 1, x0:x1 + 1].any(axis=0)
    xs = np.nonzero(cols)[0]
    if len(xs) == 0:
        xs = np.array([0, x1 - x0])
    lo, hi = int(xs.min()), int(xs.max())
    gaps, run = [], None                      # (start, end) of empty column runs inside the glyph span
    for i in range(lo, hi + 1):
        if not cols[i] and run is None:
            run = i
        elif cols[i] and run is not None:
            gaps.append((run, i)); run = None
    gaps = [gp for gp in gaps if gp[1] - gp[0] >= max(2, 0.25 * glyph_h)]
    total = sum(len(p) for p in parts)
    cuts, acc = [], 0
    for p in parts[:-1]:
        acc += len(p)
        want = lo + (hi - lo) * acc / total
        best = min((gp for gp in gaps if gp not in cuts), key=lambda gp: abs((gp[0] + gp[1]) / 2 - want), default=None)
        if best is None:
            cuts.append((int(want), int(want)))
        else:
            cuts.append(best)
    cuts.sort()
    edges = [lo] + [c for gp in cuts for c in gp] + [hi + 1]
    boxes = []
    for i in range(len(parts)):
        a, b = edges[2 * i], edges[2 * i + 1]
        seg = np.nonzero(cols[a:b])[0]
        if len(seg):
            a, b = a + int(seg.min()), a + int(seg.max()) + 1
        # stray line pieces in the text row make a segment too wide: trim its outer side
        max_w = int(len(parts[i]) * glyph_h * 1.2)
        if b - a > max_w:
            if i == 0:
                a = b - max_w
            elif i == len(parts) - 1:
                b = a + max_w
            else:
                mid = (a + b) // 2; a, b = mid - max_w // 2, mid + max_w // 2
        boxes.append([x0 + a, y0, x0 + b, y1])
    return boxes


def _slant_band(lab, box, shape):
    """Mask of the rotated text strip inside an axis-aligned box of a slanted label (|angle| > 4).
    For a box W x H holding a strip of length l and thickness t at angle a:
    W = l cos a + t sin a, H = l sin a + t cos a."""
    ang = float(lab.get('angle') or 0.0)
    if abs(ang) <= 4:
        return None
    x0, y0, x1, y1 = box
    Wb, Hb = x1 - x0, y1 - y0
    a = math.radians(abs(ang)); ca, sa = math.cos(a), math.sin(a)
    den = ca * ca - sa * sa
    if den < 0.3:
        return None
    ln = (Wb * ca - Hb * sa) / den
    t = (Hb * ca - Wb * sa) / den
    if ln <= 0 or t <= 0:
        return None
    if lab.get('glyph_h'):              # loose OCR quads overstate the thickness; glyph size is tighter
        t = min(t, 1.2 * float(lab['glyph_h']))
    rect = ((x0 + x1) / 2.0, (y0 + y1) / 2.0), (ln + 2 * S, t + 2 * S), ang
    pts = cv2.boxPoints(rect).astype(np.int32)
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [pts], 1)
    return m


def erase_text(img, labels):
    """Wipe label boxes but keep: underlines (long horizontal runs), straight lines crossing the
    box, and leader lines entering the box from one side (up to the glyph core)."""
    H, W = img.shape
    ink = (img < 200).astype(np.uint8)
    hline = _line_mask(ink)
    ink_nh = ink & (1 - hline)
    out = img.copy()
    pad = 16 * S
    for lab in labels:
        x0, y0, x1, y1 = lab['box']
        x0, y0, x1, y1 = max(x0 - S, 0), max(y0 - S, 0), min(x1 + S, W - 1), min(y1 + S, H - 1)
        # glyph core = union of glyph-like components inside the box
        glyph_h, _, _ = _glyph_band(ink_nh, (x0, y0, x1, y1))
        g = _glyph_components(ink_nh, (x0, y0, x1, y1), glyph_h)
        g[:, :x0] = 0; g[:, x1 + 1:] = 0
        ys, xs = np.nonzero(g)
        core = (int(xs.min()) - S, int(ys.min()) - S, int(xs.max()) + S, int(ys.max()) + S) if len(xs) else (x0, y0, x1, y1)

        X0, Y0, X1, Y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, W - 1), min(y1 + pad, H - 1)
        ring = ink * 255
        ring[y0:y1 + 1, x0:x1 + 1] = 0
        segs = cv2.HoughLinesP(np.ascontiguousarray(ring[Y0:Y1 + 1, X0:X1 + 1]).astype(np.uint8), 1, np.pi / 180,
                               threshold=8 * S, minLineLength=8 * S, maxLineGap=2 * S)
        keep = np.zeros_like(ink)
        crossing = entering = 0
        for sx0, sy0, sx1, sy1 in (segs.reshape(-1, 4) if segs is not None else []):
            ax, ay, bx, by = sx0 + X0, sy0 + Y0, sx1 + X0, sy1 + Y0
            ang = abs(math.degrees(math.atan2(by - ay, bx - ax))) % 180
            flat = ang < 20 or ang > 160    # exact horizontals are kept by hline; tilted ones are not
            L = math.hypot(bx - ax, by - ay); ux, uy = (bx - ax) / L, (by - ay) / L
            before, after, inside = [], [], []
            for t in np.arange(-80 * S, 80 * S, 0.5):
                ix, iy = int(round(ax + ux * t)), int(round(ay + uy * t))
                if not (X0 <= ix <= X1 and Y0 <= iy <= Y1):
                    continue
                if x0 <= ix <= x1 and y0 <= iy <= y1:
                    inside.append((ix, iy)); continue
                hit = bool(ink[max(iy - S, 0):iy + S + 1, max(ix - S, 0):ix + S + 1].any())
                (before if not inside else after).append(hit)
            if not inside:
                continue
            ok_b = len(before) >= 6 * S and np.mean(before) > 0.7
            ok_a = len(after) >= 6 * S and np.mean(after) > 0.7
            if ok_b and ok_a:
                cv2.line(keep, inside[0], inside[-1], 1, 2 * S); crossing += 1
            elif flat:
                continue                    # a flat stub ending in the box is more likely a glyph stroke
            elif ok_b or ok_a:
                # leader entering from one side: follow ink inward, stop at the glyph core
                path = inside if ok_b else inside[::-1]
                last = None
                for ix, iy in path:
                    if core[0] <= ix <= core[2] and core[1] <= iy <= core[3]:
                        break
                    if not ink[max(iy - S, 0):iy + S + 1, max(ix - S, 0):ix + S + 1].any():
                        break
                    last = (ix, iy)
                if last is not None:
                    cv2.line(keep, path[0], last, 1, 2 * S); entering += 1
        box = np.zeros_like(ink); box[y0:y1 + 1, x0:x1 + 1] = 1
        band = _slant_band(lab, (x0, y0, x1, y1), (H, W))
        if band is not None:            # rotated label: its axis box is mostly map, wipe the text strip only
            box &= band
        wipe = (box > 0) & (keep == 0) & (hline == 0)
        # text placement box: glyph core (leader stubs must not widen it), unless glyphs fused with
        # lines were left out of the core -> fall back to all wiped ink, capped at the expected width
        exp_w = len(lab.get('text', '')) * glyph_h * 1.05
        ys, xs = np.nonzero(wipe & (img < 160))
        ink_box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else [x0, y0, x1, y1]
        if core != (x0, y0, x1, y1) and (core[2] - core[0] - 2 * S) >= 0.7 * exp_w:
            lab['tight'] = [core[0] + S, core[1] + S, core[2] - S + 1, core[3] - S + 1]
        else:
            cx, half = (ink_box[0] + ink_box[2]) / 2, min(ink_box[2] - ink_box[0], exp_w * 1.1) / 2
            ty0, ty1 = (core[1] + S, core[3] - S + 1) if core != (x0, y0, x1, y1) else (ink_box[1], ink_box[3])
            lab['tight'] = [int(cx - half), ty0, int(cx + half), ty1]
        out[wipe] = 255
        cX0, cY0, cX1, cY1 = max(x0 - 3 * S, 0), max(y0 - 3 * S, 0), min(x1 + 4 * S, W), min(y1 + 4 * S, H)
        sub = (out[cY0:cY1, cX0:cX1] < 200).astype(np.uint8)
        n, cc, st, _ = cv2.connectedComponentsWithStats(sub, 8)
        for i in range(1, n):
            if max(st[i][2], st[i][3]) < 12 * S and st[i][4] < 40 * S * S:
                if band is not None and not band[cY0:cY1, cX0:cX1][cc == i].any():
                    continue            # a crumb of a line/numeral outside the slanted text strip
                comp = cc == i
                if box[cY0:cY1, cX0:cX1][comp].mean() < 0.5:
                    continue            # mostly outside the label: a map symbol (city dot) beside it
                out[cY0:cY1, cX0:cX1][comp] = 255
        lab['lines_kept'] = {'crossing': crossing, 'entering': entering}
    return _heal_crossings(img, out, labels)


def _heal_crossings(img, out, labels):
    """Re-join lines the wipe cut through. Hough misses lines that cross a big or slanted label
    (a fault line through a rotated region name lost 50 px). For every pair of long line stubs that
    end at a label box, are collinear with each other, and whose joining path was dark in the
    original, restore the original pixels along that path."""
    H, W = img.shape
    ink0 = img < 200
    ink1 = (out < 200).astype(np.uint8)
    for lab in labels:
        x0, y0, x1, y1 = lab['box']
        pad = 10 * S
        X0, Y0, X1, Y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, W), min(y1 + pad, H)
        sub = ink1[Y0:Y1, X0:X1]
        n, cc, st, _ = cv2.connectedComponentsWithStats(sub, 8)
        near = np.zeros_like(sub, bool)
        # erase_text also drops crumbs up to 4S outside the box, so stubs can end that far out
        bx0, by0, bx1, by1 = x0 - X0 - 6 * S, y0 - Y0 - 6 * S, x1 - X0 + 6 * S, y1 - Y0 + 6 * S
        near[max(by0, 0):by1, max(bx0, 0):bx1] = True
        stubs = []
        for i in range(1, n):
            sx, sy, sw, sh = st[i][:4]
            leaves = sx == 0 or sy == 0 or sx + sw == sub.shape[1] or sy + sh == sub.shape[0]
            if max(sw, sh) < 4 * S or not leaves:    # a line stub runs on out of the window
                continue
            ys, xs = np.nonzero((cc == i) & near)
            if len(xs) < 2 * S:
                continue
            ys_all, xs_all = np.nonzero(cc == i)
            cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
            d = np.hypot(xs - cx, ys - cy)
            tip = np.array([xs[d.argmin()], ys[d.argmin()]], float)
            dd = np.hypot(xs_all - tip[0], ys_all - tip[1])
            sel = dd < 6 * S                # fault lines curve: fit the last few px only
            if sel.sum() < 3 * S:
                continue
            vx, vy, _, _ = cv2.fitLine(np.stack([xs_all[sel], ys_all[sel]], 1).astype(np.float32),
                                       cv2.DIST_L2, 0, 0.01, 0.01).ravel()
            stubs.append((tip, np.array([vx, vy])))
        for a in range(len(stubs)):
            for b in range(a + 1, len(stubs)):
                (ta, va), (tb, vb) = stubs[a], stubs[b]
                gap = tb - ta
                L = float(np.hypot(*gap))
                if L < 2 * S or L > float(np.hypot(x1 - x0, y1 - y0)) + 8 * S:
                    continue
                u = gap / L
                if abs(u @ va) < 0.9 or abs(u @ vb) < 0.9:
                    continue
                path = np.zeros((H, W), np.uint8)
                p0 = (int(ta[0]) + X0, int(ta[1]) + Y0); p1 = (int(tb[0]) + X0, int(tb[1]) + Y0)
                cv2.line(path, p0, p1, 1, 2 * S + 1)
                line1 = np.zeros((H, W), np.uint8); cv2.line(line1, p0, p1, 1, 1)
                on = line1 > 0
                dark = cv2.dilate(ink0.astype(np.uint8), np.ones((2 * S + 1, 2 * S + 1), np.uint8))[on]
                if dark.mean() < 0.8:
                    continue
                m = (path > 0) & ink0
                out[m] = img[m]
    return out


def measure_text_h(src_bgr, box):
    """Height of the glyphs inside a label box, in source px (0 if unmeasurable). Neutral dark pixels
    only (colour fills excluded), long horizontal / vertical rules removed. Used to sanity-check the
    text size: OCR box height is no substitute (glyph/box ratio measured 0.45-1.0 across drawings)."""
    x0, y0, x1, y1 = [int(v) for v in box]
    sub = src_bgr[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1]
    if sub.size == 0 or sub.shape[0] < 4 or sub.shape[1] < 4:
        return 0
    lab = cv2.cvtColor(sub, cv2.COLOR_BGR2LAB).astype(np.int16)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    ink = ((cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY) < 170) & (chroma < 25)).astype(np.uint8)
    hh, ww = ink.shape
    rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, max(3, int(ww * 0.6))), np.uint8)) | \
        cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((max(3, int(hh * 0.9)), 1), np.uint8))
    rows = np.where((ink & (1 - rules)).sum(axis=1) > 0)[0]
    # first..last inked row. Taking only the heaviest run of rows (to skip a neighbouring legend row)
    # was tried: CJK glyphs have multi-row gaps at low resolution and real heights came out 20-30%
    # short. Over-tall measurements are bounded by the width cap in fit_horizontal_text instead.
    return int(rows[-1] - rows[0] + 1) if len(rows) else 0


BOLD_REL = 1.3      # source stroke / predicted regular 新宋体 stroke above this -> bold


def _stroke_width(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    per = sum(cv2.arcLength(c, True) for c in cnts)
    return 2.0 * float(mask.sum()) / per if per > 0 else 0.0


def blur_factor(src_gray, sr_gray, label_boxes_2x):
    """How much the upscaled source inflates stroke widths (blur, JPEG, interpolation): long non-text
    linework measured on the upscaled source vs on the sharp SR image. Measured 1.3 clean, 1.43 at 0.6x,
    1.91 blurred - without it a blurred drawing's labels got 2.4x too much faux bold."""
    h2, w2 = sr_gray.shape
    up = cv2.resize(src_gray, (w2, h2), interpolation=cv2.INTER_CUBIC)
    notext = np.ones((h2, w2), bool)
    for x0, y0, x1, y1 in label_boxes_2x:
        notext[max(y0 - 4, 0):y1 + 5, max(x0 - 4, 0):x1 + 5] = False

    def lines(g):
        m = ((g < min(160, int(np.percentile(g, 50)) - 40)) & notext).astype(np.uint8)
        n, lbl, st, _ = cv2.connectedComponentsWithStats(m, 8)
        big = [i for i in range(1, n) if max(st[i][2], st[i][3]) >= 40 * S]
        return np.isin(lbl, big).astype(np.uint8) if big else None

    a, b = lines(up), lines(sr_gray)
    if a is None or b is None:
        return 1.0
    sa, sb = _stroke_width(a), _stroke_width(b)
    return float(np.clip(sa / sb, 1.0, 3.0)) if sb > 0 else 1.0


def is_bold(src_gray, tight_2x, text_h, blur=1.0):
    """Whether a label is visibly heavier than the regular CJK text font. Stroke width is estimated as
    2*area/perimeter of the glyph ink (source upscaled to working scale, long rules removed) relative
    to the glyph height; the regular font's ratio was measured on rendered output as
    0.0056 + 1.3/h (h in working px, i.e. a ~1.3 px stroke). Test drawings measured 1.3-2.2x that."""
    th = float(text_h) * S
    if th < 6:
        return False
    x0, y0, x1, y1 = [int(v) for v in tight_2x]
    up = cv2.resize(src_gray[max(y0 // S - 1, 0):y1 // S + 2, max(x0 // S - 1, 0):x1 // S + 2], None,
                    fx=S, fy=S, interpolation=cv2.INTER_CUBIC)
    if up.size == 0:
        return False
    m = (up < min(160, int(np.percentile(up, 50)) - 40)).astype(np.uint8)
    hh, ww = m.shape
    rules = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((1, max(3, int(ww * 0.6))), np.uint8)) |         cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((max(3, int(hh * 0.9)), 1), np.uint8))
    m = m & (1 - rules)
    cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    per = sum(cv2.arcLength(c, True) for c in cnts)
    if per < 1 or m.sum() < 5:
        return False
    stroke = 2.0 * m.sum() / per / max(blur, 1.0)      # working px, source blur removed
    regular = 0.0056 * th + 1.3
    if stroke <= BOLD_REL * regular:
        return 0.0
    # CorelDRAW does not synthesize bold for 新宋体 (Story.Bold has no visible effect): the caller thickens
    # with a same-colour outline of this width, capped so glyph counters do not fill in
    return round(float(min(stroke - regular, 0.12 * th)), 2)


def extract_fine_marks(gray, max_thick=1.5, max_area=40, max_dim=24, radius=10, min_neighbours=3):
    """Stipple dots and short hatch strokes that PowerTRACE silently drops (1-2 px wide at working
    scale: in a 0.6x drawing it lost 120 of 215 components, all area <= 30 px, thickness <= 1.4).
    Candidates are thin small components that sit in a dense patch of similar marks - fills and
    hatching come in patches, isolated specks are noise and are left alone. Returns the dot / stroke
    geometry (working px) and the image with those marks removed, so they are not traced twice."""
    ink = (gray < 128).astype(np.uint8)
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    n, lbl, st, cen = cv2.connectedComponentsWithStats(ink, 8)
    cand = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a > max_area or max(w, h) > max_dim:
            continue
        comp = lbl[y:y + h, x:x + w] == i
        if dist[y:y + h, x:x + w][comp].max() > max_thick:
            continue
        cand.append(i)
    if not cand:
        return {'dots': [], 'strokes': []}, gray
    pts = cen[cand]
    r = radius * S
    keep = []
    for j, i in enumerate(cand):
        d = np.hypot(pts[:, 0] - pts[j, 0], pts[:, 1] - pts[j, 1])
        if int((d <= r).sum()) - 1 >= min_neighbours:
            keep.append(i)
    dots, strokes = [], []
    out = gray.copy()
    for i in keep:
        x, y, w, h, a = st[i]
        comp = lbl[y:y + h, x:x + w] == i
        ys, xs = np.nonzero(comp)
        xs = xs + x + 0.5
        ys = ys + y + 0.5
        length = float(max(w, h))
        if length >= 4 and length * length / max(a, 1) >= 2.5:
            # elongated: principal axis through the pixels, endpoints at the extreme projections
            c = np.array([xs.mean(), ys.mean()])
            cov = np.cov(np.vstack([xs - c[0], ys - c[1]]))
            evals, evecs = np.linalg.eigh(cov)
            v = evecs[:, np.argmax(evals)]
            proj = (xs - c[0]) * v[0] + (ys - c[1]) * v[1]
            p0, p1 = c + v * proj.min(), c + v * proj.max()
            span = float(np.hypot(*(p1 - p0))) or 1.0
            strokes.append([round(float(p0[0]), 1), round(float(p0[1]), 1), round(float(p1[0]), 1),
                            round(float(p1[1]), 1), round(max(1.2, a / (span + 1.0)), 2)])
        else:
            dots.append([round(float(xs.mean()), 1), round(float(ys.mean()), 1),
                         round(max(1.0, float(np.sqrt(a / np.pi))), 2)])
        out[y:y + h, x:x + w][comp] = 255
    return {'dots': dots, 'strokes': strokes}, out


LATIN_FONTS = {'Times New Roman': 'C:/Windows/Fonts/times.ttf', 'Arial': 'C:/Windows/Fonts/arial.ttf'}


def detect_latin_font(src_gray, labels, default='Times New Roman'):
    """Proportional font for labels without CJK characters. The CJK font (新宋体) is monospaced: "0.5" came
    out a character-width wider than the source and ran into axis ticks. Render each Latin label in each
    candidate and correlate with the source glyphs; majority vote, default when unclear. (The same test
    does NOT tell 宋体 from 黑体 reliably at drawing resolutions, so the CJK font is left alone.)"""
    from PIL import Image, ImageDraw, ImageFont
    fonts = {n: p for n, p in LATIN_FONTS.items() if os.path.exists(p)}
    if not fonts:
        return default
    votes = {n: 0 for n in fonts}
    for lab in labels:
        text = str(lab['text']).replace('|', '')
        if len(text) < 2 or _CJK.search(text) or abs(float(lab.get('angle', 0) or 0)) > 4:
            continue
        x0, y0, x1, y1 = [int(v) // S for v in lab.get('tight', lab['box'])]
        crop = src_gray[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1]
        if crop.size == 0 or crop.shape[0] < 6:
            continue
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        a = cv2.GaussianBlur((crop < np.percentile(crop, 50) - 30).astype(np.float32), (3, 3), 0)
        best, best_s = None, -1.0
        for name, path in fonts.items():
            f = ImageFont.truetype(path, 64)
            bb = f.getbbox(text)
            im = Image.new('L', (bb[2] - bb[0] + 4, bb[3] - bb[1] + 4), 255)
            ImageDraw.Draw(im).text((2 - bb[0], 2 - bb[1]), text, font=f, fill=0)
            r = cv2.resize(np.array(im), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
            b = cv2.GaussianBlur((r < 128).astype(np.float32), (3, 3), 0)
            sc = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))
            if sc > best_s:
                best, best_s = name, sc
        votes[best] += 1
    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    if ranked[0][1] >= 2 and (len(ranked) == 1 or ranked[0][1] >= 1.5 * ranked[1][1]):
        return ranked[0][0]
    return default


def finalize(out_dir, confirmed):
    """confirmed: list of {text, box:[x0,y0,x1,y1] in SOURCE px} (or a path to such json)."""
    if isinstance(confirmed, str):
        confirmed = json.load(open(confirmed, encoding='utf-8'))
    src_bgr = imread(os.path.join(out_dir, 'src.png'), cv2.IMREAD_COLOR)
    sr2 = imread(os.path.join(out_dir, 'sr2.png'))
    h, w = imread(os.path.join(out_dir, 'src.png')).shape
    ink = (sr2 < 200).astype(np.uint8)
    ink_nh = ink & (1 - _rule_mask(ink))
    labels = []
    for c in confirmed:
        text = str(c['text']).strip()
        if not text:
            continue
        box = [int(v) * S for v in c['box']]
        box = [max(box[0], 0), max(box[1], 0), min(box[2], w * S - 1), min(box[3], h * S - 1)]
        # OCR often reads adjacent labels as one line: "集流环|测井仪器车|测井仪器板" splits the box
        parts = [p.strip() for p in text.split('|') if p.strip()]
        if len(parts) > 1:
            for p, sub in zip(parts, _split_box(ink_nh, box, parts)):
                labels.append({'text': p, 'box': sub})
        else:
            ang = float(c.get('angle', 0.0) or 0.0)
            # a slanted / vertical OCR box already spans the whole label: growing it sideways would
            # follow neighbouring ink along the wrong axis
            grown = box if abs(ang) > 4 else expand_to_text(ink_nh, box, len(text))
            labels.append({'text': text, 'box': grown, 'angle': ang})
    for lab in labels:
        lab['text_h'] = measure_text_h(src_bgr, [v // S for v in lab['box']])
    clean = erase_text(sr2, labels)
    for lab in labels:
        bx0, by0, bx1, by1 = lab['box']
        tx0, ty0, tx1, ty1 = lab['tight']
        # faint / blurred glyphs barely enter the ink mask: the located box collapses onto a fragment
        # and the text was placed tiny and off-centre (滚筒). The confirmed box is the better estimate.
        if (tx1 - tx0) < 0.45 * (bx1 - bx0) or (ty1 - ty0) < 0.35 * (by1 - by0):
            lab['tight'] = [bx0, by0, bx1, by1]
    src_gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    blur = blur_factor(src_gray, sr2, [l['box'] for l in labels])
    for lab in labels:
        lab['bold_px'] = (is_bold(src_gray, lab.get('tight', lab['box']), lab.get('text_h', 0), blur)
                          if abs(lab.get('angle', 0.0)) <= 4 else 0.0)
    marks, clean = extract_fine_marks(clean)
    imwrite(os.path.join(out_dir, 'sr2_no_text.png'), clean)
    meta = {'src_size': [w, h], 'scale': S, 'labels': labels, 'marks': marks,
            'latin_font': detect_latin_font(src_gray, labels)}
    with open(os.path.join(out_dir, 'labels.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return meta


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        m = prepare(sys.argv[2], sys.argv[3])
        print(json.dumps(m, ensure_ascii=False))
    elif sys.argv[1] == 'finalize':
        m = finalize(sys.argv[2], sys.argv[3])
        print(json.dumps(m, ensure_ascii=False))
    else:
        raise SystemExit(__doc__)
