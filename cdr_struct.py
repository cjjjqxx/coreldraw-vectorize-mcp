"""Structural check of a vectorize result: are the same ELEMENTS there, at the same width and colour?

Coverage / precision / colour recall (cdr_selfcheck.measure) stay near 1.0 while a drawing is visibly
wrong: the flowchart that lost every arrowhead scored coverage 0.999, and the tiled-trace bug that painted
whole boxes black was reported as "no issues". A missing arrowhead is 40 px out of a million, and a black
box is content where content belongs, so neither shows up in an area ratio.

What is measured instead, over a palette taken from the SOURCE (never from the pipeline's own clustering):

  marks       small elements (symbols, arrowheads, dashes, dots, swatch borders): each source mark must be
              reproduced somewhere within 2 px, and each result mark must correspond to something
  regions     large areas: compared as AREAS per colour (IoU), so a region split by a line counts once
  width       drawn width of thin marks, source vs result - catches "the 2 px border came out 0.9 px"
  colour      Lab distance per reproduced mark
  edges       75th percentile distance from a source edge to the nearest result edge: did boundaries move

Used by cdr_selfcheck (reported as issues on every build) and by regress/struct_eval.py.
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




def ink_balance(src, res, valid, tol=40, near=2):
    """(extra_share, missing_share): pixels the result paints much darker than the source and vice versa.

    The mark test asks "is there ink where the source has ink", which a box painted solid black passes
    everywhere. This asks the other question - is there ink where the source has NONE - and that is what
    a filled-in box, a flooded region or a lost line actually looks like."""
    s_l = _lab(src)[..., 0]
    r_l = _lab(res)[..., 0]
    k = np.ones((2 * near + 1, 2 * near + 1), np.uint8)
    darker = (r_l < s_l - tol) & valid                 # result darker than the source
    lighter = (s_l < r_l - tol) & valid                # source darker than the result
    s_dark_near = cv2.dilate((s_l < np.percentile(s_l, 50) - tol).astype(np.uint8), k) > 0
    r_dark_near = cv2.dilate((r_l < np.percentile(r_l, 50) - tol).astype(np.uint8), k) > 0
    extra = float((darker & ~s_dark_near).mean())
    missing = float((lighter & ~r_dark_near).mean())
    return round(extra, 4), round(missing, 4)


def check(src, res, label_boxes=(), min_mark_area=40):
    """Structural comparison of two rasters of the same drawing. Returns the metrics dict; `res` is
    resized to the source. label_boxes (source px) are excluded - text is checked separately."""
    h, w = src.shape[:2]
    if res.shape[:2] != (h, w):
        res = cv2.resize(res, (w, h), interpolation=cv2.INTER_AREA)
    valid = np.ones((h, w), bool)
    for x0, y0, x1, y1 in label_boxes:
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
    r_iou, _rows = region_iou(src, res, pal, valid)
    w_err = []
    for e, c in zip(s_small, s_cov):
        if c < 0.5 or e['width'] > 6:
            continue
        x, y, bw, bh = e['bbox']
        cands = [r for r in r_small if abs(r['bbox'][0] + r['bbox'][2] / 2 - (x + bw / 2)) < 10
                 and abs(r['bbox'][1] + r['bbox'][3] / 2 - (y + bh / 2)) < 10]
        if cands:
            r = min(cands, key=lambda r_: abs(r_['area'] - e['area']))
            if not (0.3 <= r['area'] / max(e['area'], 1) <= 3.5):
                continue                           # not the same mark: a rule matched to a whole row
            w_err.append((abs(e['width'] - r['width']), abs(e['width'] - r['width']) / max(e['width'], 1.0)))
    s_lab, r_lab = _lab(src), _lab(res)
    dE = []
    for e, c in zip(s_small, s_cov):
        if c < 0.5 or e['area'] < 8:
            continue
        x, y, bw, bh = e['bbox']
        m = e['mask'] > 0
        dE.append(float(np.linalg.norm(np.median(s_lab[y:y + bh, x:x + bw][m], axis=0)
                                       - np.median(r_lab[y:y + bh, x:x + bw][m], axis=0))))
    ink_extra, ink_missing = ink_balance(src, res, valid)
    big_missing = [e for e in missing if e['area'] >= min_mark_area]
    return {
        'src_marks': len(s_small), 'res_marks': len(r_small),
        'mark_recall': round((len(s_small) - len(missing)) / max(len(s_small), 1), 3),
        'mark_precision': round((len(r_small) - len(extra)) / max(len(r_small), 1), 3),
        'missing': len(missing), 'missing_significant': len(big_missing), 'extra': len(extra),
        'region_iou': r_iou,
        # median of the absolute error, plus the relative one: on a table the absolute error is dominated
        # by whichever rule happened to be matched to a filled row, and says nothing about the lines
        'width_mae': round(float(np.median([a for a, _r in w_err])), 2) if w_err else None,
        'width_rel': round(float(np.median([r for _a, r in w_err])), 2) if w_err else None,
        'colour_dE_median': round(float(np.median(dE)), 1) if dE else None,
        'edge_p75_px': edge_distance(src, res, valid),
        'ink_extra_share': ink_extra, 'ink_missing_share': ink_missing,
        'src_elements': len(s_el),
        'missing_regions': [[int(v) for v in e['bbox']] for e in sorted(big_missing, key=lambda e: -e['area'])[:8]],
        'extra_regions': [[int(v) for v in e['bbox']] for e in sorted(extra, key=lambda e: -e['area'])[:8]],
    }


def issues(metrics, objects=None):
    """Turn the metrics into the issue records the calling AI acts on."""
    out = []
    m = metrics
    if m.get('missing_significant', 0) >= 3 and (m.get('mark_recall') or 1) < 0.85:
        out.append({'type': 'elements_missing', 'stage': 'graphics',
                    'regions': m.get('missing_regions', []),
                    'detail': f"{m['missing_significant']} small element(s) of the source (arrowheads, "
                              f"symbols, dashes, borders) are not in the result",
                    'hint': 'raise colors, or mode="lineart" for black symbols; these are the elements a '
                            'coverage figure cannot see'})
    if (m.get('width_rel') or 0) > 0.6 and (m.get('width_mae') or 0) > 1.0:
        out.append({'type': 'line_width', 'stage': 'graphics',
                    'detail': f"drawn line width is off by {m['width_mae']} px ({int(100 * m['width_rel'])}%)",
                    'hint': 'outlines / strokes were measured wrong: check the rim width of the region '
                            'model or the stroke width of the layer model'})
    if (m.get('ink_extra_share') or 0) > 0.01:
        out.append({'type': 'ink_extra', 'stage': 'graphics',
                    'detail': f"{round(100 * m['ink_extra_share'], 1)}% of the page is painted much darker "
                              f"than the source",
                    'hint': 'something was filled in that should not be: an open path closed by the tracer, '
                            'a region painted over its neighbour, or a mask inverted'})
    if (m.get('ink_missing_share') or 0) > 0.01:
        out.append({'type': 'ink_missing', 'stage': 'graphics',
                    'detail': f"{round(100 * m['ink_missing_share'], 1)}% of the source's dark content is "
                              f"not drawn",
                    'hint': 'linework was dropped: check trace detail / the line layers'})
    if (m.get('region_iou') or 1) < 0.85:
        out.append({'type': 'regions_shifted', 'stage': 'graphics',
                    'detail': f"colour regions overlap the source by only {m['region_iou']}",
                    'hint': 'a region is the wrong colour, shifted, or painted over another one'})
    # a boundary shift only counts when the regions disagree too: on a chart full of thin curves the edge
    # distance is dominated by curve detail that has no counterpart, not by boundaries that moved
    if (m.get('edge_p75_px') or 0) > 5.0 and (m.get('region_iou') or 1) < 0.92:
        out.append({'type': 'boundary_shift', 'stage': 'graphics',
                    'detail': f"region boundaries sit {m['edge_p75_px']} px away from the source edges",
                    'hint': 'boundaries wandered: smoothing too strong, or the partition edge is wrong'})
    if objects and m.get('src_elements') and objects > 3 * m['src_elements']:
        out.append({'type': 'fragmented', 'stage': 'graphics',
                    'detail': f"{objects} objects for about {m['src_elements']} source elements",
                    'hint': 'the drawing was traced into fragments; on a flat-colour diagram the region '
                            'model (CDR_FLAT=1) is the fix, otherwise raise smoothing / lower detail'})
    return out
