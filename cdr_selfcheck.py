"""Graphic self-check of a vectorize build: result.png against the MCP's own input (src.png).

No ground truth is needed, so it runs on every call. It measures, outside the label boxes (text is
re-created and checked separately):
  coverage       share of source content (non-paper pixels) the result reproduces (2 px tolerance)
  precision      share of result content that has source content nearby
  colour_recall  share of chromatic source pixels rendered with a similar hue
and turns shortfalls into issues the calling AI can act on, with a diff image
(red = in the source but missing, blue = drawn but not in the source).
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

PAPER_MIN = 225          # min channel at/above this is paper
CHROMA = 25              # OpenCV Lab chroma above this is "coloured"


def _imread(p, flag=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(p, np.uint8), flag)


def measure(src, res, label_boxes):
    h, w = src.shape[:2]
    res = cv2.resize(res, (w, h), interpolation=cv2.INTER_AREA)
    zone = np.zeros((h, w), np.uint8)
    for x0, y0, x1, y1 in label_boxes:
        zone[max(int(y0) - 3, 0):int(y1) + 4, max(int(x0) - 3, 0):int(x1) + 4] = 1
    valid = zone == 0
    s_ink = (src.min(axis=2) < PAPER_MIN) & valid
    r_ink = (res.min(axis=2) < PAPER_MIN) & valid
    k5 = np.ones((5, 5), np.uint8)
    r_near = cv2.dilate(r_ink.astype(np.uint8), k5) > 0
    s_near = cv2.dilate(s_ink.astype(np.uint8), k5) > 0
    # a pixel only counts as missing / extra if the colours really differ: a pale fill a few levels either
    # side of PAPER_MIN (source #fee2e2 vs rendered #fddfe0) was reported as a whole extra box
    diff = np.abs(src.astype(np.int16) - res.astype(np.int16)).max(axis=2) > 30
    diff_near = cv2.erode(diff.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    missing = s_ink & ~r_near & diff_near
    extra = r_ink & ~s_near & diff_near
    coverage = 1.0 - missing.sum() / max(s_ink.sum(), 1)
    precision = 1.0 - extra.sum() / max(r_ink.sum(), 1)

    s_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    r_lab = cv2.cvtColor(res, cv2.COLOR_BGR2LAB).astype(np.float32)
    s_ab, r_ab = s_lab[..., 1:] - 128, r_lab[..., 1:] - 128
    s_col = (np.hypot(s_ab[..., 0], s_ab[..., 1]) > CHROMA) & valid
    # compare against the most similar result pixel in a 5x5 neighbourhood: edges are allowed to shift
    ok = np.zeros((h, w), bool)
    r_ch = np.hypot(r_ab[..., 0], r_ab[..., 1]) > CHROMA * 0.6
    s_hue = np.arctan2(s_ab[..., 1], s_ab[..., 0])
    r_hue = np.arctan2(r_ab[..., 1], r_ab[..., 0])
    for dy in (-2, 0, 2):
        for dx in (-2, 0, 2):
            rh = np.roll(np.roll(r_hue, dy, 0), dx, 1)
            rc = np.roll(np.roll(r_ch, dy, 0), dx, 1)
            ok |= rc & (np.abs(np.angle(np.exp(1j * (s_hue - rh)))) < np.radians(30))
    colour_missing = s_col & ~ok
    colour_recall = 1.0 - colour_missing.sum() / max(s_col.sum(), 1) if s_col.sum() > 200 else None
    return {'coverage': round(float(coverage), 3), 'precision': round(float(precision), 3),
            'colour_recall': None if colour_recall is None else round(float(colour_recall), 3),
            'source_px': int(s_ink.sum())}, missing, extra, colour_missing, res


def lost_elements(src, res, valid, min_area=12, max_frac=0.02):
    """Compact solid elements of the source (dots, symbols, small patches) that the result does not
    reproduce in a similar colour. Area-based coverage cannot see these: one lost highlight dot in a chart
    is 0.1% of the content. Returns boxes in source px."""
    h, w = src.shape[:2]
    res = cv2.resize(res, (w, h), interpolation=cv2.INTER_AREA)
    content = ((src.min(axis=2) < PAPER_MIN) & valid).astype(np.uint8)
    solid = cv2.morphologyEx(content, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(solid, 8)
    close = (np.abs(src.astype(np.int16) - res.astype(np.int16)).max(axis=2) <= 60).astype(np.uint8)
    close = cv2.dilate(close, np.ones((3, 3), np.uint8)) > 0          # 1 px registration slack
    lost = []
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if a < min_area or a > max_frac * h * w or max(bw, bh) > 8 * min(bw, bh):
            continue                                                     # tiny, huge, or a line
        comp = lbl[y:y + bh, x:x + bw] == i
        if close[y:y + bh, x:x + bw][comp].mean() < 0.4:
            lost.append([int(x), int(y), int(x + bw), int(y + bh)])
    return lost


def _regions(mask, min_area, limit=8):
    m = cv2.dilate(mask.astype(np.uint8), np.ones((7, 7), np.uint8))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, 8)
    regs = sorted(((int(st[i][4]), [int(st[i][0]), int(st[i][1]), int(st[i][0] + st[i][2]), int(st[i][1] + st[i][3])])
                   for i in range(1, n) if st[i][4] >= min_area), reverse=True)
    return [r[1] for r in regs[:limit]]


def _norm(t):
    table = str.maketrans({'～': '~', '〜': '~', '－': '-', '—': '-', '（': '(', '）': ')', '，': ',', '、': ',',
                           '：': ':', '？': '?', '！': '!', '；': ';', 'º': '°', '˚': '°', '＜': '<', '　': ''})
    return ''.join(str(t).translate(table).split())


def _cer(pred, gt):
    a, b = _norm(pred), _norm(gt)
    if not b:
        return 0.0
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[len(b)] / len(b)


def text_check(work_dir, labels, scale):
    """Read the rendered result back and compare with what was placed. Catches text that is placed but
    renders wrong: overlapping neighbours, wildly wrong size, missing glyphs. Horizontal labels only.
    A label is reported only if OCR reads it on the source but not on the result, so OCR's own limits
    (tiny glyphs, neighbours read as one line) do not produce false alarms."""
    from rapidocr_onnxruntime import RapidOCR
    eng = RapidOCR(text_score=0.3)

    def read_all(img, to_src):
        out, _ = eng(img, use_cls=False)
        dets = []
        for quad, t, sc in out or []:
            q = np.array(quad, np.float32) / to_src
            dets.append(((q[:, 0].min() + q[:, 0].max()) / 2, (q[:, 1].min() + q[:, 1].max()) / 2, t))
        return dets

    # result.png is exported at twice the SOURCE size, which equals the working scale only when the
    # super-resolution factor happens to be 2. On a large drawing (SR skipped, scale 1) every rendered
    # detection then landed at double the coordinates of the label boxes, matched nothing, and every
    # label was reported as "renders as nothing": 61 of 104 on the geology map, all of them false.
    res_img = _imread(os.path.join(work_dir, 'result.png'))
    src = _imread(os.path.join(work_dir, 'src.png'))
    res_to_src = (res_img.shape[1] / float(src.shape[1])) if src is not None and src.shape[1] else scale
    res_dets = read_all(res_img, res_to_src)
    src_dets = read_all(cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC), scale)

    def read_at(dets, box):
        x0, y0, x1, y1 = box
        gx, gy = 0.25 * (x1 - x0) + 2, 0.25 * (y1 - y0) + 2
        hits = sorted([d for d in dets if x0 - gx <= d[0] <= x1 + gx and y0 - gy <= d[1] <= y1 + gy],
                      key=lambda d: d[0])
        return ''.join(d[2] for d in hits)

    bad = []
    others = sorted({_norm(l['text'].replace('|', '')) for l in labels}, key=len, reverse=True)
    for lab in labels:
        if abs(float(lab.get('angle', 0) or 0)) > 4:
            continue
        text = lab['text'].replace('|', '')
        box = [v / scale for v in lab['box']]
        src_read = read_at(src_dets, box)
        e_src = _cer(src_read, text)
        if e_src > 0.34 and _norm(text) not in _norm(src_read):
            continue                                 # OCR cannot read this label even on the source
        res_read = read_at(res_dets, box)

        def own(read):
            # OCR merges neighbouring labels into one line ("300450km"): drop other labels' text first
            r = _norm(read)
            for other in others:
                if other and other not in _norm(text) and other in r:   # never strip part of this label
                    r = r.replace(other, '', 1)
            return r
        e_src_own, e_res_own = _cer(own(src_read), text), _cer(own(res_read), text)
        # flag only when the rendered label reads clearly WORSE than the source did: a ghost glyph
        # ("质量合格??"), a missing or garbled character. A result that reads better than a degraded
        # source ("8.4°" -> "84°") is fine.
        if e_res_own > e_src_own + 0.1:
            bad.append({'text': lab['text'], 'rendered_read': res_read, 'source_read': src_read,
                        'box': [int(v) for v in box]})
    return bad


def _label_boxes_scaled(work_dir):
    """Label boxes in WORKING pixels (already multiplied by the scale) and that scale."""
    for name in ('layers.json', 'labels.json'):
        p = os.path.join(work_dir, name)
        if os.path.exists(p):
            meta = json.load(open(p, encoding='utf-8'))
            return [l['box'] for l in meta.get('labels', [])], float(meta.get('scale', 2))
    return [], 2.0


def wipe_spill(work_dir, limit=8):
    """Pixels the label wipe removed OUTSIDE every label box.

    erase_text() wipes label boxes to white before PowerTRACE runs, so the graphics are traced from an
    image the text was cut out of. A box that is too large - or one that was given for a piece of
    linework rather than for text - therefore deletes linework no later step can bring back, and the
    final check can only report it as an unexplained "missing graphics". This measures the damage
    directly so the caller learns that a box is destructive. Boxes are padded by 16 working px, which
    is the wipe's own margin.
    """
    a = os.path.join(work_dir, 'sr2.png')
    b = os.path.join(work_dir, 'sr2_no_text.png')
    if not (os.path.exists(a) and os.path.exists(b)):
        # colour mode wipes the ink layer instead of a single composite image
        a = os.path.join(work_dir, 'ink_before_wipe.png')
        b = os.path.join(work_dir, 'ink_no_text.png')
    if not (os.path.exists(a) and os.path.exists(b)):
        return None
    before = _imread(a, cv2.IMREAD_GRAYSCALE)
    after = _imread(b, cv2.IMREAD_GRAYSCALE)
    if before.shape != after.shape:
        return None
    h, w = before.shape[:2]
    ink_before = before < 200
    total_ink = int(ink_before.sum())
    if total_ink == 0:
        return None
    boxes, S = _label_boxes_scaled(work_dir)
    pad = int(16 * S)
    zone = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in boxes:
        zone[max(int(y0) - pad, 0):int(y1) + pad + 1, max(int(x0) - pad, 0):int(x1) + pad + 1] = True
    spill = ink_before & ~zone & (after >= 200)
    n = int(spill.sum())
    if n < max(60, 0.002 * total_ink):
        return None
    return {'px': n, 'share': round(n / total_ink, 4),
            'regions': _regions(spill, max(12, int(h * w * 0.00005)), limit)}


def graphics_self_check(work_dir, trace_stats=None):
    """CHECKPOINT run BEFORE any text is placed: graphics_only.png against the text-free image that
    was traced (sr2_no_text.png).

    The final check compares result.png against src.png, where a lost line and a text object sitting on
    top of a line produce the very same symptom - a hole at those coordinates - so the caller cannot
    tell whether to change mode/colors or to fix a label. Here the page holds nothing but traced
    graphics, so every shortfall is a tracing fault and the fix is mode/colors/image quality, never a
    label edit. Returns None when the checkpoint image is missing.
    """
    res_p = os.path.join(work_dir, 'graphics_only.png')
    if not os.path.exists(res_p):
        return None
    # The image that was actually traced. Lineart traces one text-free composite, so that can be
    # compared over the whole page. Colour mode traces one mask per palette colour and never
    # materialises such a composite, so it falls back to src.png with the label zones excluded - no
    # text has been placed yet, so everything outside those zones is still a pure graphics comparison.
    src_p = os.path.join(work_dir, 'sr2_no_text.png')
    if os.path.exists(src_p):
        src = _imread(src_p)
        zone_boxes: list = []
    else:
        src_p = os.path.join(work_dir, 'src.png')
        if not os.path.exists(src_p):
            return None
        src = _imread(src_p)
        wb, scale = _label_boxes_scaled(work_dir)
        zone_boxes = [[v / scale for v in b] for b in wb]
    res = _imread(res_p)
    metrics, missing, extra, colour_missing, res_s = measure(src, res, zone_boxes)
    h, w = src.shape[:2]
    valid = np.ones((h, w), bool)
    for x0, y0, x1, y1 in zone_boxes:
        valid[max(int(y0) - 3, 0):int(y1) + 4, max(int(x0) - 3, 0):int(x1) + 4] = False
    lost = lost_elements(src, res, valid)
    metrics['lost_elements'] = len(lost)
    min_area = max(30, int(h * w * 0.0005))
    issues = []
    # Structural check: coverage says 0.999 while every arrowhead is missing or a box is painted black,
    # because those are tiny areas or "content where content belongs". cdr_struct compares ELEMENTS.
    try:
        import cdr_struct
        struct = cdr_struct.check(src, res, zone_boxes)
        metrics['struct'] = {k: v for k, v in struct.items() if not k.endswith('regions')}
        n_obj = sum(int(st.get('curves') or 0) + int(st.get('lines') or 0) for st in (trace_stats or []))
        issues.extend(cdr_struct.issues(struct, objects=n_obj or None))
    except Exception as exc:  # noqa: BLE001 - never fail a build over the check itself
        metrics['struct_error'] = repr(exc)[:200]
    for st in trace_stats or []:
        if st.get('kind') in ('color', 'ink', 'ink_faint', 'grid') and st.get('curves') == 0:
            issues.append({'type': 'trace_failed', 'stage': 'graphics', 'layer': st.get('layer'),
                           'colour': st.get('color'),
                           'detail': 'PowerTRACE returned no curves for this layer; its content is missing',
                           'hint': 'rebuild; if it persists try mode="lineart" or report the image size'})
    if metrics['coverage'] < 0.9:
        issues.append({'type': 'graphics_missing', 'stage': 'graphics',
                       'detail': f"only {metrics['coverage']:.0%} of the traced source is reproduced, "
                                 'measured before any text was placed (red in diff_png)',
                       'regions': _regions(missing, min_area),
                       'hint': 'a tracing fault, not a text problem: colour content lost -> '
                               'mode="color" / colors=16; thin lines lost -> mode="lineart"'})
    if lost:
        issues.append({'type': 'lost_elements', 'stage': 'graphics', 'regions': lost[:20],
                       'detail': f'{len(lost)} small solid element(s) are missing or wrongly coloured '
                                 'before any text was placed',
                       'hint': 'a lost colour usually needs a larger colors value, a lost black '
                               'symbol mode="lineart"'})
    if metrics['precision'] < 0.9:
        issues.append({'type': 'graphics_extra', 'stage': 'graphics',
                       'detail': f"{1 - metrics['precision']:.0%} of the drawn graphics have no "
                                 'counterpart in the traced source, before any text was placed',
                       'regions': _regions(extra, min_area),
                       'hint': 'stray strokes from the trace itself: rebuild; a persistent spike '
                               'usually means the source needs cropping'})
    if metrics['colour_recall'] is not None and metrics['colour_recall'] < 0.85:
        issues.append({'type': 'colour_mismatch', 'stage': 'graphics',
                       'detail': f"only {metrics['colour_recall']:.0%} of the coloured source area is "
                                 'drawn in a matching colour',
                       'regions': _regions(colour_missing, min_area),
                       'hint': 'rebuild with mode="color" and a larger colors value'})
    spill = wipe_spill(work_dir)
    if spill:
        issues.append({'type': 'wipe_spill', 'stage': 'labels', 'regions': spill['regions'],
                       'detail': f"{spill['px']} px of linework outside every label box were wiped as "
                                 'text before tracing, so they cannot appear in the result',
                       'hint': 'a label box is too large, or was given for linework rather than text: '
                               'shrink or drop that label and rebuild (editing only its text will not help)'})
    vis = src.copy()
    vis[missing] = (0, 0, 255)
    vis[extra] = (255, 0, 0)
    vis[colour_missing & ~missing] = (0, 200, 255)
    diff_png = os.path.join(work_dir, 'graphics_check.png')
    cv2.imencode('.png', vis)[1].tofile(diff_png)
    return {'metrics': metrics, 'issues': issues, 'diff_png': diff_png, 'spill': spill}


def self_check(work_dir, trace_stats=None):
    """Returns {'metrics', 'issues', 'diff_png'}; issues is [] when nothing looks wrong."""
    src = _imread(os.path.join(work_dir, 'src.png'))
    res = _imread(os.path.join(work_dir, 'result.png'))
    boxes, labels, s = [], [], 2
    for name in ('layers.json', 'labels.json'):
        p = os.path.join(work_dir, name)
        if os.path.exists(p):
            meta = json.load(open(p, encoding='utf-8'))
            s = meta.get('scale', 2)
            labels = meta.get('labels', [])
            boxes = [[v / s for v in l['box']] for l in labels]
            break
    metrics, missing, extra, colour_missing, res_s = measure(src, res, boxes)
    h, w = src.shape[:2]
    zone = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in boxes:
        zone[max(int(y0) - 3, 0):int(y1) + 4, max(int(x0) - 3, 0):int(x1) + 4] = True
    lost = lost_elements(src, res, ~zone)
    metrics['lost_elements'] = len(lost)
    min_area = max(30, int(h * w * 0.0005))
    issues = []
    for st in trace_stats or []:
        if st.get('kind') in ('color', 'ink', 'ink_faint', 'grid') and st.get('curves') == 0:
            issues.append({'type': 'trace_failed', 'stage': 'graphics', 'layer': st.get('layer'),
                           'colour': st.get('color'),
                           'detail': 'PowerTRACE returned no curves for this layer; its content is missing',
                           'hint': 'rebuild; if it persists try mode="lineart" or report the image size'})
    if metrics['coverage'] < 0.9:
        issues.append({'type': 'graphics_missing', 'stage': 'graphics',
                       'detail': f"only {metrics['coverage']:.0%} of the source "
                       'graphics are reproduced (red in diff_png)', 'regions': _regions(missing, min_area),
                       'hint': 'look at diff_png; colour content lost -> try mode="color" / colors=16; '
                               'thin lines lost -> mode="lineart"'})
    if lost:
        issues.append({'type': 'lost_elements', 'stage': 'graphics', 'regions': lost[:20],
                       'detail': f'{len(lost)} small solid element(s) of the source (dots, symbols, patches) are '
                                 'missing or have the wrong colour in the result',
                       'hint': 'compare these regions on result.png; a lost colour usually needs a larger colors '
                               'value, a lost black symbol mode="lineart"'})
    if metrics['precision'] < 0.9:
        issues.append({'type': 'graphics_extra', 'stage': 'graphics',
                       'detail': f"{1 - metrics['precision']:.0%} of the drawn "
                       'graphics have no counterpart in the source (blue in diff_png)',
                       'regions': _regions(extra, min_area),
                       'hint': 'often leftovers of text that OCR missed or misplaced: fix labels and rebuild'})
    if metrics['colour_recall'] is not None and metrics['colour_recall'] < 0.85:
        issues.append({'type': 'colour_mismatch', 'stage': 'graphics',
                       'detail': f"only {metrics['colour_recall']:.0%} of the "
                       'coloured source area is drawn in a matching colour', 'regions': _regions(colour_missing, min_area),
                       'hint': 'rebuild with mode="color" and a larger colors value'})
    try:
        bad_text = text_check(work_dir, labels, s) if labels else []
    except Exception:  # noqa: BLE001 - OCR unavailable: skip the text part
        bad_text = []
    if bad_text:
        issues.append({'type': 'text_mismatch', 'stage': 'text', 'labels': bad_text[:20],
                       'detail': f'{len(bad_text)} placed label(s) do not read back as placed (overlap, size or '
                                 'missing glyphs)',
                       'hint': 'check these boxes on result.png; fix the text/box in labels and rebuild, or '
                               'try another font'})
    metrics['text_readback_failed'] = len(bad_text)
    vis = src.copy()
    vis[missing] = (0, 0, 255)
    vis[extra] = (255, 0, 0)
    vis[colour_missing & ~missing] = (0, 200, 255)
    diff_png = os.path.join(work_dir, 'self_check.png')
    cv2.imencode('.png', vis)[1].tofile(diff_png)
    return {'metrics': metrics, 'issues': issues, 'diff_png': diff_png}


if __name__ == '__main__':
    import sys
    print(json.dumps(self_check(sys.argv[1]), ensure_ascii=False, indent=1))
