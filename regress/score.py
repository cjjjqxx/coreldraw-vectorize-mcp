"""Score a vectorize result against the CLEAN reference image (whatever degraded input was used).

Everything is measured at reference scale, outside the ground-truth label boxes for graphics and
inside them for text:
  colour regions  per-colour F1 (1 px tolerance) + mean CIE76 colour error
  ink / grid      F1 of dark-neutral linework and light-neutral graticule
  blobs           extra ink specks, ink symbols / colour patches present in the reference but lost
  text            OCR of the result vs ground truth; only labels OCR can read on the reference count

usage: python score.py <reference.png> <result.png> <gt_labels.json> [out_dir]
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

PAPER, INK, GRID, EDGE = 0, 1, 2, -1
COLOR0 = 10


def imread(p, flag=cv2.IMREAD_COLOR):
    img = cv2.imdecode(np.fromfile(p, np.uint8), flag)
    if img is None:
        raise ValueError(f'cannot read {p}')
    return img


def lab_f(bgr):
    """true CIE Lab (L 0-100)."""
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def ref_palette(bgr, k=8):
    lab = lab_f(bgr)
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    sel = (chroma >= 18) & (bgr.min(axis=2) < 235)
    data = lab[sel].reshape(-1, 3)
    if len(data) < 50:
        return np.zeros((0, 3), np.float32)
    if len(data) > 80000:
        data = data[np.random.default_rng(0).choice(len(data), 80000, replace=False)]
    k = min(k, len(data))
    _, lbl, cen = cv2.kmeans(data.astype(np.float32), k, None,
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5), 4, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(lbl.ravel(), minlength=k)
    keep = []
    for i in np.argsort(-counts):
        if counts[i] < 0.01 * len(data):
            continue
        if all(np.linalg.norm(cen[i] - cen[j]) > 12 for j in keep):
            keep.append(i)
    return cen[keep]


def classify(bgr, palette):
    lab = lab_f(bgr)
    L = lab[..., 0]
    chroma = np.hypot(lab[..., 1], lab[..., 2])
    out = np.full(L.shape, EDGE, np.int16)
    # paper >= 238: graticule rules are drawn at ~222-232, a 228 cut split them between paper and grid
    out[bgr.min(axis=2) >= 238] = PAPER
    neutral = chroma < 12
    out[neutral & (L < 55)] = INK
    out[neutral & (L >= 72) & (bgr.min(axis=2) < 238)] = GRID
    col = (chroma >= 18) & (bgr.min(axis=2) < 240)
    if len(palette):
        d = np.stack([np.hypot(lab[..., 1] - c[1], lab[..., 2] - c[2]) + 0.3 * np.abs(L - c[0]) for c in palette])
        out[col] = COLOR0 + d.argmin(axis=0)[col]
    return out


def f1(ref_m, out_m, valid, tol=1):
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    r = ref_m & valid
    o = out_m & valid
    nr, no = int(r.sum()), int(o.sum())
    if nr == 0 and no == 0:
        return None
    prec = float((o & (cv2.dilate(r.astype(np.uint8), k) > 0)).sum()) / max(no, 1)
    rec = float((r & (cv2.dilate(o.astype(np.uint8), k) > 0)).sum()) / max(nr, 1)
    return {'f1': round(2 * prec * rec / max(prec + rec, 1e-9), 3), 'precision': round(prec, 3),
            'recall': round(rec, 3), 'ref_px': nr, 'out_px': no}


def components(mask, min_area):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [(i, st[i]) for i in range(1, n) if st[i][4] >= min_area], lab


def lost_components(ref_m, out_m, valid, min_area, max_area=None, cover=0.3, tol=2):
    """reference components that the output does not reproduce."""
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    out_d = cv2.dilate((out_m & valid).astype(np.uint8), k) > 0
    comps, lab = components(ref_m & valid, min_area)
    lost = []
    for i, st in comps:
        x, y, w, h, a = [int(v) for v in st]
        if max_area and a > max_area:
            continue
        sel = lab[y:y + h, x:x + w] == i
        if out_d[y:y + h, x:x + w][sel].mean() < cover:
            lost.append([x, y, x + w, y + h])
    return lost


def norm_text(s):
    table = str.maketrans({'～': '~', '〜': '~', '－': '-', '—': '-', '（': '(', '）': ')', '，': ',', '、': ',',
                           'º': '°', '˚': '°', '＜': '<', '　': ''})
    return ''.join(s.translate(table).split())


def cer(pred, gt):
    a, b = norm_text(pred), norm_text(gt)
    if not b:
        return 0.0
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
    return d[len(b)] / len(b)


_OCR = None


def ocr(bgr, scale):
    """RapidOCR detections as (text, score, box in reference px)."""
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR(text_score=0.3)
    res, _ = _OCR(bgr)
    dets = []
    for quad, text, score in res or []:
        q = np.array(quad, np.float32) / scale
        dets.append({'text': text, 'score': float(score), 'box': [float(q[:, 0].min()), float(q[:, 1].min()),
                                                                     float(q[:, 0].max()), float(q[:, 1].max())]})
    return dets


def read_label(dets, box):
    """Concatenate detections whose centre falls in the (slightly grown) label box."""
    x0, y0, x1, y1 = box
    gx, gy = 0.25 * (x1 - x0) + 2, 0.25 * (y1 - y0) + 2
    hits = [d for d in dets if x0 - gx <= (d['box'][0] + d['box'][2]) / 2 <= x1 + gx
            and y0 - gy <= (d['box'][1] + d['box'][3]) / 2 <= y1 + gy]
    hits.sort(key=lambda d: (d['box'][0] + d['box'][2]) / 2)
    return ''.join(d['text'] for d in hits), hits


def text_score(ref_bgr, out_bgr, gt):
    up = 3 if ref_bgr.shape[1] < 1200 else 2
    H, W = ref_bgr.shape[:2]
    ref_d = ocr(cv2.resize(ref_bgr, (W * up, H * up), interpolation=cv2.INTER_CUBIC), up)
    interp = cv2.INTER_AREA if out_bgr.shape[1] > W * up else cv2.INTER_CUBIC
    out_d = ocr(cv2.resize(out_bgr, (W * up, H * up), interpolation=interp), up)
    rows = []
    for g in gt:
        text = g['text'].replace('|', '')
        box = g['box']
        r_txt, _ = read_label(ref_d, box)
        o_txt, o_hits = read_label(out_d, box)
        row = {'text': text, 'box': box, 'ref_read': r_txt, 'out_read': o_txt,
               'ref_cer': round(cer(r_txt, text), 2), 'out_cer': round(cer(o_txt, text), 2)}
        if len(o_hits) == 1 and abs(float(g.get('angle', 0))) <= 4 and '|' not in g['text']:
            hb = o_hits[0]['box']
            row['size_ratio'] = round((hb[3] - hb[1]) / max(box[3] - box[1], 1), 2)
            row['offset'] = round(float(np.hypot((hb[0] + hb[2] - box[0] - box[2]) / 2,
                                                 (hb[1] + hb[3] - box[1] - box[3]) / 2)) / max(box[3] - box[1], 1), 2)
        rows.append(row)
    readable = [r for r in rows if r['ref_cer'] <= 0.34]
    ok = [r for r in readable if r['out_cer'] <= 0.34]
    sizes = [abs(r['size_ratio'] - 1) for r in readable if 'size_ratio' in r]
    return {'readable': len(readable), 'ok': len(ok), 'text_ok': round(len(ok) / max(len(readable), 1), 3),
            'size_err_median': round(float(np.median(sizes)), 2) if sizes else None,
            'failed': [{'text': r['text'], 'box': r['box'], 'out_read': r['out_read'], 'cer': r['out_cer']}
                       for r in readable if r['out_cer'] > 0.34],
            'rows': rows}


def score(ref_path, out_path, gt_path, out_dir=None):
    ref = imread(ref_path)
    out_full = imread(out_path)
    H, W = ref.shape[:2]
    out = cv2.resize(out_full, (W, H), interpolation=cv2.INTER_AREA)
    gt = json.load(open(gt_path, encoding='utf-8'))
    text_zone = np.zeros((H, W), np.uint8)
    for g in gt:
        x0, y0, x1, y1 = g['box']
        text_zone[max(y0 - 2, 0):y1 + 3, max(x0 - 2, 0):x1 + 3] = 1
    valid = text_zone == 0
    pal = ref_palette(ref)
    rc, oc = classify(ref, pal), classify(out, pal)

    colors = []
    for j in range(len(pal)):
        r = f1(rc == COLOR0 + j, oc == COLOR0 + j, valid)
        if r:
            lab = pal[j]
            r['ref_lab'] = [round(float(v), 1) for v in lab]
            colors.append(r)
    both = (rc >= COLOR0) & (oc >= COLOR0) & valid
    de = float(np.linalg.norm(lab_f(ref)[both] - lab_f(out)[both], axis=1).mean()) if both.any() else None
    tot = sum(c['ref_px'] for c in colors)
    color_f1 = round(sum(c['f1'] * c['ref_px'] for c in colors) / tot, 3) if tot else None

    ink = f1(rc == INK, oc == INK, valid)
    # anti-aliased flanks of black lines / colour patches are light neutral too: they are not grid
    halo = cv2.dilate(((rc == INK) | (rc >= COLOR0)).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    grid_ref = (rc == GRID) & valid & ~halo
    # drawings without a graticule only have scattered light-grey pixels (hatching, AA): no grid score
    grid = f1(rc == GRID, oc == GRID, valid & ~halo, tol=2) if grid_ref.mean() > 0.002 else None
    ink_ref, ink_out = (rc == INK), (oc == INK)
    col_ref, col_out = rc >= COLOR0, oc >= COLOR0
    extra_ink = lost_components(ink_out, ink_ref | (cv2.dilate((rc == EDGE).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0),
                                valid, min_area=3, cover=0.2)
    lost_ink = lost_components(ink_ref, ink_out, valid, min_area=6, max_area=400)
    lost_col = lost_components(col_ref, col_out, valid, min_area=8)
    txt = text_score(ref, out_full, gt)

    metrics = {
        'color_f1': color_f1, 'color_dE': round(de, 1) if de is not None else None,
        'ink_f1': ink['f1'] if ink else None, 'grid_f1': grid['f1'] if grid else None,
        'extra_ink_blobs': len(extra_ink), 'lost_ink_symbols': len(lost_ink), 'lost_color_patches': len(lost_col),
        'text_ok': txt['text_ok'], 'text_readable': txt['readable'], 'size_err_median': txt['size_err_median'],
    }
    report = {'metrics': metrics, 'colors': colors, 'ink': ink, 'grid': grid,
              'issues': {'extra_ink': extra_ink, 'lost_ink': lost_ink, 'lost_color': lost_col,
                         'text_failed': txt['failed']},
              'text_rows': txt['rows']}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'score.json'), 'w', encoding='utf-8') as fh:
            json.dump(report, fh, ensure_ascii=False, indent=1)
        sheet = np.hstack([ref, np.full((H, 6, 3), (0, 0, 255), np.uint8), out.copy()])
        ov = sheet[:, W + 6:]
        for bx in extra_ink:
            cv2.rectangle(ov, (bx[0] - 2, bx[1] - 2), (bx[2] + 2, bx[3] + 2), (255, 0, 0), 1)       # blue: extra
        for bx in lost_ink + lost_col:
            cv2.rectangle(ov, (bx[0] - 2, bx[1] - 2), (bx[2] + 2, bx[3] + 2), (0, 0, 255), 1)       # red: lost
        for t in txt['failed']:
            b = t['box']
            cv2.rectangle(ov, (b[0] - 1, b[1] - 1), (b[2] + 1, b[3] + 1), (0, 160, 255), 2)       # orange: text
        cv2.imencode('.png', sheet)[1].tofile(os.path.join(out_dir, 'sheet.png'))
    return report


if __name__ == '__main__':
    rep = score(*sys.argv[1:5])
    print(json.dumps(rep['metrics'], ensure_ascii=False))
