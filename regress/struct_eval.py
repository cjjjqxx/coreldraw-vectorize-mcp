"""Structural evaluation of a vectorize build: does the result contain the same ELEMENTS as the source?

The self-check in the MCP measures pixel coverage, precision and colour recall. Those stay near 1.0 while
the drawing is visibly wrong: a flowchart whose every arrowhead is missing scored coverage 0.999, and the
tiled-trace bug that painted whole boxes black was reported as "no issues" for weeks. Coverage cannot see
it, because a missing arrowhead is 40 px out of a million and a black box is "content where content is".

Here both images are decomposed into ELEMENTS and matched one by one:

  elements      per palette colour (a palette derived from the SOURCE, independent of the MCP's own),
                connected components above a minimum area - colour patches, strokes, dots, symbols
  recall        source elements that have a matching element in the result (IoU over the pair)
  precision     result elements that correspond to something in the source
  width_mae     mean |width difference| over matched THIN elements (lines: area per skeleton pixel) -
                this is what catches "the 2 px border came out 0.9 px"
  colour_dE     Lab distance between matched elements' colours
  edge_p90      90th percentile distance from a source region edge to the nearest result region edge:
                how far the boundaries wandered (wobbly edges, shifted bands)
  objects       curves + strokes actually created in CorelDRAW, from _coreldraw.log, against the number
                of source elements: a drawing of 200 elements traced as 3000 curves is fragmented

Text is excluded (the label boxes); cdr_selfcheck.text_check covers it.

usage: python struct_eval.py <work_dir> [work_dir ...]        prints a table, writes struct_eval.json
"""
from __future__ import annotations

import json
import os
import re
import sys

import cv2
import numpy as np

MIN_AREA = 12            # source px; below this a component is noise, not an element
PAPER_MIN = 235


def imread(p, flag=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), flag)


def _lab(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def source_palette(src, max_colours=24):
    """Palette of the SOURCE, used for both images so the comparison does not inherit the MCP's own
    clustering. k-means in Lab over non-paper pixels, close centres merged."""
    lab = _lab(src)
    small = cv2.resize(lab, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST).reshape(-1, 3)
    keep = small[:, 0] < 253
    data = small[keep] if keep.sum() > 500 else small
    if len(data) > 60000:
        rng = np.random.default_rng(0)
        data = data[rng.choice(len(data), 60000, replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, _, cen = cv2.kmeans(data.astype(np.float32), max_colours, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    merged = []
    for c in sorted(cen, key=lambda c: -c[0]):
        if all(np.linalg.norm(c - m) > 12 for m in merged):
            merged.append(c)
    return np.array(merged, np.float32)


def assign(lab, palette):
    d = np.stack([np.linalg.norm(lab - c, axis=2) for c in palette], axis=0)
    return d.argmin(axis=0).astype(np.int32), d.min(axis=0)


def elements(bgr, palette, valid, min_area=MIN_AREA):
    """[(colour index, mask, area, width, colour)] for every component of every palette colour."""
    lab = _lab(bgr)
    idx, dist = assign(lab, palette)
    out = []
    for j in range(len(palette)):
        m = ((idx == j) & (dist < 22) & valid).astype(np.uint8)
        if not m.any():
            continue
        # JPEG speckle quantises into thousands of 12-40 px blobs; an ELEMENT is either solid (survives a
        # 3x3 erosion) or a real run of a line (skeleton at least 10 px long). Without this the metric
        # reported 1000+ "missing elements" on a drawing that looked right.
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, lb, st, _ = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            if st[i, 4] < min_area:
                continue
            x, y, w, h, a = st[i]
            # the crop only: a full-size mask per element ran to hundreds of MB on a real drawing
            comp = (lb[y:y + h, x:x + w] == i).astype(np.uint8)
            pad = cv2.copyMakeBorder(comp, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            try:
                skel = cv2.ximgproc.thinning(pad * 255)
                slen = int((skel > 0).sum())
                width = float(a) / float(max(slen, 1))
            except Exception:  # noqa: BLE001 - no contrib module: width from the thickest point
                slen = max(w, h)
                width = 2.0 * float(cv2.distanceTransform(pad, cv2.DIST_L2, 3).max())
            solid = int(cv2.erode(pad, np.ones((3, 3), np.uint8)).sum())
            if solid < 4 and slen < 10:
                continue                           # neither a body nor a run: speckle
            sub_lab = lab[y:y + h, x:x + w][comp > 0]
            out.append({'colour': int(j), 'bbox': (int(x), int(y), int(w), int(h)), 'area': int(a),
                        'width': round(width, 2), 'mask': comp,
                        'lab': np.median(sub_lab, axis=0).astype(np.float32)})
    return out


SMALL_MAX = 2500         # source px: above this an element is a REGION, below it a mark/symbol/line


def region_iou(src, res, palette, valid):
    """Per palette colour, IoU of the whole colour's mask. Regions are compared as AREAS, not as
    components: a background split by three dashed faults is one region in the result and four
    components in the source, and component matching called that four misses plus one extra."""
    s_idx, s_d = assign(_lab(src), palette)
    r_idx, r_d = assign(_lab(res), palette)
    rows = []
    for j in range(len(palette)):
        sm = (s_idx == j) & (s_d < 22) & valid
        rm = (r_idx == j) & (r_d < 22) & valid
        if int(sm.sum()) < 200:
            continue
        inter = int((sm & rm).sum())
        union = int((sm | rm).sum())
        rows.append({'colour': j, 'src_px': int(sm.sum()), 'iou': inter / max(union, 1)})
    if not rows:
        return None, []
    tot = sum(r['src_px'] for r in rows)
    return round(sum(r['iou'] * r['src_px'] for r in rows) / max(tot, 1), 3), rows


def covered(marks, other_img, palette, valid, thr=0.5, dE=30.0):
    """A mark counts as reproduced when at least `thr` of its pixels carry a similar colour in the other
    image. One-to-one matching was too brittle: a legend swatch whose fill grew over its border, or a star
    redrawn two pixels fatter, counted as one miss plus one extra although both were clearly drawn."""
    lab_o = _lab(other_img)
    H, W = lab_o.shape[:2]
    k5 = np.ones((5, 5), np.uint8)                # +-2 px tolerance, as in the MCP's own self-check
    out = []
    for e in marks:
        x, y, bw, bh = e['bbox']
        m = e['mask'] > 0
        x0, y0 = max(x - 3, 0), max(y - 3, 0)
        x1, y1 = min(x + bw + 3, W), min(y + bh + 3, H)
        crop = lab_o[y0:y1, x0:x1]
        ref = e['lab']
        if float(ref[0]) < 150:
            # dark ink: an anti-aliased 2 px line is mid-grey in the source and solid black in the result,
            # which is more than dE apart although the line is plainly there. Match on "as dark, same hue".
            near = ((crop[..., 0] <= ref[0] + 40) &
                    (np.linalg.norm(crop[..., 1:] - ref[1:], axis=2) < 25)).astype(np.uint8)
        else:
            near = (np.linalg.norm(crop - ref, axis=2) < dE).astype(np.uint8)
        near = cv2.dilate(near, k5) > 0
        sub = near[y - y0:y - y0 + bh, x - x0:x - x0 + bw][m]
        out.append(float(sub.mean()) if sub.size else 0.0)
    return out


def edge_distance(src, res, valid):
    """75th percentile distance from a source edge pixel to the nearest result edge pixel (px).

    The 90th percentile was dominated by source detail that has no counterpart at all (a stipple field
    read 106 px on a map whose boundaries were in fact within a pixel), which says nothing about whether
    the boundaries moved. The 75th percentile measures the bulk of the edges."""
    e_s = (cv2.Canny(cv2.GaussianBlur(src, (3, 3), 0), 40, 120) > 0) & valid
    e_r = (cv2.Canny(cv2.GaussianBlur(res, (3, 3), 0), 40, 120) > 0) & valid
    if not e_s.any() or not e_r.any():
        return None
    dt = cv2.distanceTransform((~e_r).astype(np.uint8), cv2.DIST_L2, 3)
    d = dt[e_s]
    return round(float(np.percentile(d, 75)), 2)


def object_count(work_dir):
    """curves + strokes actually created, from the CorelDRAW step's log."""
    p = os.path.join(work_dir, '_coreldraw.log')
    if not os.path.exists(p):
        return None
    t = open(p, encoding='utf-8', errors='ignore').read()
    curves = sum(int(x) for x in re.findall(r'"curves": (\d+)', t))
    lines = sum(int(x) for x in re.findall(r'"lines": (\d+)', t))
    return curves + lines


def label_boxes(work_dir):
    try:
        meta = json.load(open(os.path.join(work_dir, 'layers.json'), encoding='utf-8'))
    except Exception:  # noqa: BLE001
        return [], 1
    S = meta.get('scale') or 1
    out = []
    for lb in meta.get('labels', []):
        b = lb.get('box') or lb.get('tight')
        if b:
            out.append([v / S for v in b])
    return out, S


def evaluate(work_dir):
    src = imread(os.path.join(work_dir, 'src.png'))
    res = imread(os.path.join(work_dir, 'result.png'))
    if src is None or res is None:
        return {'error': 'missing src.png or result.png'}
    h, w = src.shape[:2]
    res = cv2.resize(res, (w, h), interpolation=cv2.INTER_AREA)
    boxes, _S = label_boxes(work_dir)
    valid = np.ones((h, w), bool)
    for x0, y0, x1, y1 in boxes:                  # text is checked separately
        valid[max(int(y0) - 3, 0):int(y1) + 4, max(int(x0) - 3, 0):int(x1) + 4] = False
    pal = source_palette(src)
    s_el = elements(src, pal, valid)
    r_el = elements(res, pal, valid)
    s_small = [e for e in s_el if e['area'] <= SMALL_MAX]
    r_small = [e for e in r_el if e['area'] <= SMALL_MAX]
    s_cov = covered(s_small, res, pal, valid)
    r_cov = covered(r_small, src, pal, valid)
    missing = [e for e, c in zip(s_small, s_cov) if c < 0.5]
    extra = [e for e, c in zip(r_small, r_cov) if c < 0.5]
    n_ok = len(s_small) - len(missing)
    r_iou, _rows = region_iou(src, res, pal, valid)
    # width: compare each reproduced thin mark with the result mark that covers it
    w_err = []
    for e, c in zip(s_small, s_cov):
        if c < 0.5 or e['width'] > 6:
            continue
        x, y, bw, bh = e['bbox']
        cands = [r for r in r_small if abs(r['bbox'][0] + r['bbox'][2] / 2 - (x + bw / 2)) < 10
                 and abs(r['bbox'][1] + r['bbox'][3] / 2 - (y + bh / 2)) < 10]
        if cands:
            w_err.append(abs(e['width'] - min(cands, key=lambda r: abs(r['area'] - e['area']))['width']))
    s_lab, r_lab = _lab(src), _lab(res)

    def px(img, e):
        x, y, bw, bh = e['bbox']
        return img[y:y + bh, x:x + bw][e['mask'] > 0]
    dE = []
    for e, c in zip(s_small, s_cov):
        if c < 0.5 or e['area'] < 8:
            continue
        x, y, bw, bh = e['bbox']
        m = e['mask'] > 0
        dE.append(float(np.linalg.norm(np.median(s_lab[y:y + bh, x:x + bw][m], axis=0)
                                       - np.median(r_lab[y:y + bh, x:x + bw][m], axis=0))))
    n_obj = object_count(work_dir)
    out = {
        'src_marks': len(s_small), 'res_marks': len(r_small),
        'mark_recall': round(n_ok / max(len(s_small), 1), 3),
        'mark_precision': round((len(r_small) - len(extra)) / max(len(r_small), 1), 3),
        'missing': len(missing), 'extra': len(extra),
        'region_iou': r_iou,
        'width_mae': round(float(np.mean(w_err)), 2) if w_err else None,
        'colour_dE_median': round(float(np.median(dE)), 1) if dE else None,
        'edge_p75_px': edge_distance(src, res, valid),
        'objects': n_obj,
        'src_elements': len(s_el),
        'objects_per_element': round(n_obj / max(len(s_el), 1), 2) if n_obj else None,
        'worst_missing': [{'bbox': e['bbox'], 'area': e['area']} for e in sorted(missing, key=lambda e: -e['area'])[:8]],
    }
    vis = (src * 0.45 + 140).astype(np.uint8)
    for e, col in [(e, (0, 0, 255)) for e in missing] + [(e, (255, 0, 0)) for e in extra]:
        x, y, bw, bh = e['bbox']
        vis[y:y + bh, x:x + bw][e['mask'] > 0] = col
    cv2.imencode('.png', vis)[1].tofile(os.path.join(work_dir, 'struct_eval.png'))
    with open(os.path.join(work_dir, 'struct_eval.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    return out


if __name__ == '__main__':
    rows = {}
    for wd in sys.argv[1:]:
        n = os.path.normpath(wd).replace(os.sep, '/').split('/')
        name = '/'.join(n[-2:]) if len(n) > 1 else n[-1]
        rows[name] = evaluate(wd)
    cols = ['src_marks', 'mark_recall', 'mark_precision', 'missing', 'extra', 'region_iou',
            'width_mae', 'colour_dE_median', 'edge_p75_px', 'objects', 'objects_per_element']
    print(f"{'case':<28}" + ''.join(f'{c[:12]:>14}' for c in cols))
    for k, v in rows.items():
        print(f'{k:<28}' + ''.join(f'{str(v.get(c)):>14}' for c in cols))
