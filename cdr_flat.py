"""Flat-colour path: partition the drawing into regions, draw each region as one vector shape.

The layer pipeline (cdr_vectorize_color) splits the image into one mask per palette colour and traces every
mask on its own with PowerTRACE. It never knows which regions exist or which ones touch, and on a flat
colour diagram that shows: the anti-aliased pixels between two regions land in whatever cluster is
nearest and trace as hundreds of slivers, two neighbours meet along two independently traced edges that
leave white seams, a dashed line becomes one shape per dash, and any ink the text wipe misses is traced as
lines under the new text. A geological section with ~200 objects came out as 2274 curves + 753 strokes.

Here the drawing is modelled instead of traced:

  1. palette  - colours taken only from FLAT pixels (3x3 neighbourhood uniform), so transition colours
                never become palette entries in the first place
  2. classes  - a colour with solid areas is a REGION colour; one that is only ever thin is a LINE colour
  3. text     - inside each OCR box, every pixel that is not the box's own background is text (the grey
                anti-aliased rim included). A piece that runs out of the box is linework and stays.
  4. lines    - line-colour pixels become vector strokes: straight dashed runs as one dashed line each,
                the rest as chained centre lines
  5. partition- region colours seed a watershed over the image; text, line and transition pixels are not
                seeds, so the regions grow across them and meet along the colour edge. Every pixel belongs
                to exactly one region: no slivers, no gaps, nothing left under the text
  6. shapes   - one filled Bezier shape per connected region, painted outer-first (by filled area) and
                grown half a pixel so neighbours overlap instead of leaving a hairline

Used by cdr_vectorize_color.py after build_layers(): the labels (text objects) and their measurements
are kept from there; the graphic layers are replaced. Selection: flat_score() / CDR_FLAT=1|0.
"""
from __future__ import annotations

import math
import os

import cv2
import numpy as np

REGION_MIN_PX = 12        # a region component smaller than this (source px) is merged into its neighbour
FLAT_STD = 3.0            # 3x3 Lab std below this: a flat pixel (inside a region, not on an edge)
SEED_DE = 10.0            # a flat pixel within this Lab distance of a palette colour seeds that colour
MERGE_DE = 9.0            # palette colours closer than this are one colour


def _lab(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def flatness(lab):
    """Per-pixel 3x3 standard deviation of Lab (max over channels)."""
    out = np.zeros(lab.shape[:2], np.float32)
    for c in range(3):
        ch = lab[..., c]
        m = cv2.blur(ch, (3, 3))
        v = cv2.blur(ch * ch, (3, 3)) - m * m
        out = np.maximum(out, np.sqrt(np.maximum(v, 0)))
    return out


def palette(lab, flat, max_colours=32, min_px=None):
    """Colours of the flat pixels, most frequent first; close ones merged."""
    h, w = flat.shape
    min_px = min_px or max(20, int(0.00002 * h * w))
    px = lab[flat].reshape(-1, 3)
    if len(px) == 0:
        return np.zeros((0, 3), np.float32), []
    q = np.floor(px / 3.0).astype(np.int32)
    keys = (q[:, 0] << 20) | ((q[:, 1] + 64) << 10) | (q[:, 2] + 64)
    uk, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    order = np.argsort(-cnt)
    centres, counts = [], []
    for o in order:
        if cnt[o] < min_px // 4 or len(centres) >= max_colours * 3:
            break
        c = px[inv == o].mean(axis=0)
        if all(np.linalg.norm(c - e) > MERGE_DE for e in centres):
            centres.append(c)
    centres = np.array(centres, np.float32)
    if len(centres) == 0:
        return centres, []
    # refine: mean of the flat pixels within SEED_DE, drop colours with too few pixels
    d = np.stack([np.linalg.norm(px - c, axis=1) for c in centres], axis=1)
    near, dist = d.argmin(axis=1), d.min(axis=1)
    out, counts = [], []
    for j in range(len(centres)):
        sel = (near == j) & (dist < SEED_DE)
        n = int(sel.sum())
        if n >= min_px:
            out.append(px[sel].mean(axis=0))
            counts.append(n)
    order = np.argsort(-np.array(counts))
    out = np.array(out, np.float32)[order][:max_colours] if out else np.zeros((0, 3), np.float32)
    return out, [counts[i] for i in order][:max_colours]


def nearest(lab, centres):
    """Index of and Lab distance to the nearest centre for every pixel."""
    h, w = lab.shape[:2]
    flat = lab.reshape(-1, 3)
    best = np.full(len(flat), 1e9, np.float32)
    idx = np.zeros(len(flat), np.int32)
    for j, c in enumerate(centres):
        d = np.linalg.norm(flat - c, axis=1)
        m = d < best
        best[m] = d[m]
        idx[m] = j
    return idx.reshape(h, w), best.reshape(h, w)


def flat_score(src_bgr):
    """(is_flat, stats). A flat-colour drawing: a few colours cover nearly every flat pixel, flat pixels
    are most of the image, and what is not flat is thin (edges, lines, text) rather than texture
    (stipple, hatching, photos - those keep the layer pipeline, which has dedicated handling for them)."""
    lab = _lab(src_bgr)
    std = flatness(lab)
    flat = std < FLAT_STD
    cen, cnt = palette(lab, flat)
    if len(cen) == 0:
        return False, {'flat_share': 0.0}
    idx, dist = nearest(lab, cen)
    covered = flat & (dist < SEED_DE)
    flat_share = float(covered.mean())
    # texture: non-flat pixels FAR from any flat pixel. A line, an edge or a text stroke is a non-flat band
    # a few px wide (the 3x3 test widens a 2 px line to ~6 px); hatching fields, photos, stipple are wide.
    nonflat = (~covered).astype(np.uint8)
    far = cv2.distanceTransform(nonflat, cv2.DIST_L2, 3) > 6.0      # dashed faults: a ~9 px band
    texture = float(far.mean())
    # many tiny isolated non-flat specks = stipple / noise
    n_cc, _, st, _ = cv2.connectedComponentsWithStats(nonflat, 8)
    small_cc = int(((st[1:, 4] >= 3) & (st[1:, 4] <= 30)).sum()) if n_cc > 1 else 0
    speck_density = small_cc / (src_bgr.shape[0] * src_bgr.shape[1] / 1e4)       # per 100x100 px
    major = int(sum(1 for c in cnt if c >= 0.0005 * src_bgr.shape[0] * src_bgr.shape[1]))
    # Scope: drawings made OF colour regions (sections, filled maps). A flowchart, chart or table is lines,
    # small marks and text on white paper; the layer pipeline handles those (arrowheads, box borders,
    # tick marks) and the region model measured worse on them (flowchart lost every arrowhead).
    Lc = lab[..., 0]
    ch = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    paper = float(((Lc > 235) & (ch < 8)).mean())
    stats = {'flat_share': round(flat_share, 3), 'colours': major, 'texture': round(texture, 4),
             'speck_density': round(speck_density, 2), 'coloured_share': round(1.0 - paper, 3)}
    ok = (flat_share >= 0.80 and texture <= 0.01 and speck_density <= 3.0 and 2 <= major <= 24
          and 1.0 - paper >= 0.5)
    return ok, stats


def _chain_centerlines(mask, S=1):
    """Centre-line polylines of a thin mask (skeleton walk, junction spurs dropped, pieces chained)."""
    import cdr_trace_skeleton as sk
    from cdr_vectorize_color import _chain_polylines
    skel = sk.skeletonize((mask > 0).astype(np.uint8) * 255)
    if int(skel.sum()) == 0:
        return [], 1.0
    width = float((mask > 0).sum()) / float(max((skel > 0).sum(), 1))
    polys = []
    for path in sk.trace_paths(skel):
        if len(path) < 4:
            continue
        ap = sk._simplify(path, 0.7).reshape(-1, 2)
        polys.append([float(v) for xy in ap for v in xy])
    return _chain_polylines(polys, 2.0), max(width, 1.0)


def _smooth_closed(pts, sigma):
    """Gaussian smoothing of a closed point loop."""
    n = len(pts)
    if n < 8 or sigma <= 0:
        return pts.astype(np.float64)
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    ext = np.concatenate([pts[-r:], pts, pts[:r]]).astype(np.float64)
    xs = np.convolve(ext[:, 0], k, mode='valid')
    ys = np.convolve(ext[:, 1], k, mode='valid')
    return np.stack([xs, ys], axis=1)


def _corners(poly, max_angle=150.0):
    n = len(poly)
    out = np.zeros(n, bool)
    for i in range(n):
        a, b, c = poly[i - 1], poly[i], poly[(i + 1) % n]
        v1, v2 = a - b, c - b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2) / (n1 * n2))))))
        out[i] = ang < max_angle
    return out


def _fit_line(pts):
    """Least-squares line through pts -> (point, unit direction, max deviation)."""
    c = pts.mean(axis=0)
    u, s_, vt = np.linalg.svd(pts - c, full_matrices=False)
    d = vt[0] / (np.linalg.norm(vt[0]) or 1.0)
    dev = float(np.abs((pts - c) @ np.array([-d[1], d[0]])).max()) if len(pts) else 0.0
    return c, d, dev


def contour_to_bezier(cnt, scale=1.0, eps=0.8, sigma=1.2, straight_dev=1.0, min_run=7.0):
    """A closed contour (N x 2, pixel corners) -> start point and cubic segments
    [(x, y, c1x, c1y, c2x, c2y), ...] in `scale` units.

    Straight edges are FITTED, not traced. Smoothing a pixel staircase and simplifying it leaves a
    wobble on every long edge and rounds off every right angle - the strata boundaries came out
    visibly crooked and the legend swatches lost their corners. Each span between two break points is
    tested against its least-squares line: a span that is straight becomes one straight segment (snapped
    to exactly horizontal / vertical when it is within 2 degrees), and where two straight spans meet the
    vertex is their INTERSECTION, which restores the sharp corner. Only spans that really curve keep a
    Catmull-Rom curve through the smoothed points.
    """
    raw = cnt.reshape(-1, 2).astype(np.float64)
    n_raw = len(raw)
    if n_raw < 6:
        return None
    # break points: a coarse simplification of the raw contour, as indices into raw
    simp = cv2.approxPolyDP(raw.astype(np.float32).reshape(-1, 1, 2), 1.2, True).reshape(-1, 2)
    if len(simp) < 3:
        return None
    idx = sorted({int(np.argmin(np.linalg.norm(raw - p, axis=1))) for p in simp})
    if len(idx) < 3:
        return None
    spans = []                                    # (i0, i1, straight, point, dir, length)
    for k in range(len(idx)):
        i0, i1 = idx[k], idx[(k + 1) % len(idx)]
        seg = raw[i0:i1 + 1] if i1 > i0 else np.vstack([raw[i0:], raw[:i1 + 1]])
        if len(seg) < 2:
            continue
        L = float(np.linalg.norm(seg[-1] - seg[0]))
        c, d, dev = _fit_line(seg) if len(seg) >= 3 else (seg.mean(axis=0),
                                                          (seg[-1] - seg[0]) / (L or 1.0), 0.0)
        straight = (L >= min_run and dev <= straight_dev) or L < 4.0   # a corner chamfer is a segment too
        if straight:                              # axis snap: drawings are full of exact H/V edges
            ang = math.degrees(math.atan2(d[1], d[0])) % 180
            if min(ang, 180 - ang) < 2.0:
                d = np.array([1.0, 0.0])
            elif abs(ang - 90) < 2.0:
                d = np.array([0.0, 1.0])
        spans.append([i0, i1, straight, c, d, L, seg])
    if not spans:
        return None
    m = len(spans)
    # vertices between consecutive spans
    verts = []
    for k in range(m):
        s0, s1 = spans[k - 1], spans[k]
        p_raw = raw[s1[0]]
        if s0[2] and s1[2]:
            d0, d1 = s0[4], s1[4]
            cross = float(d0[0] * d1[1] - d0[1] * d1[0])
            if abs(cross) > 0.08:                 # not parallel: sharp corner at the intersection
                c0, c1 = s0[3], s1[3]
                t = float(((c1[0] - c0[0]) * d1[1] - (c1[1] - c0[1]) * d1[0]) / cross)
                v = c0 + t * d0
                verts.append(v if np.linalg.norm(v - p_raw) < 6.0 else p_raw)
                continue
        if s0[2] or s1[2]:                        # one side straight: sit on that line
            sp = s0 if s0[2] else s1
            c, d = sp[3], sp[4]
            verts.append(c + d * float(np.dot(p_raw - c, d)))
            continue
        verts.append(p_raw)
    # a short span between two straight ones is the staircase of a CORNER: drop it so the two lines meet
    # at their intersection. Without this every small rectangle (the legend swatches) came out round.
    if m >= 4:
        drop = []
        for k in range(m):
            s_prev, s_cur, s_next = spans[k - 1], spans[k], spans[(k + 1) % m]
            if s_cur[5] <= 6.0 and s_prev[2] and s_next[2]:
                d0, d1 = s_prev[4], s_next[4]
                if abs(float(d0[0] * d1[1] - d0[1] * d1[0])) > 0.25:      # they really turn
                    drop.append(k)
        if drop and len(drop) < m - 2:
            spans = [sp for k, sp in enumerate(spans) if k not in drop]
            m = len(spans)
            verts = []
            for k in range(m):
                s0, s1 = spans[k - 1], spans[k]
                p_raw = raw[s1[0]]
                if s0[2] and s1[2]:
                    d0, d1 = s0[4], s1[4]
                    cross = float(d0[0] * d1[1] - d0[1] * d1[0])
                    if abs(cross) > 0.08:
                        c0, c1 = s0[3], s1[3]
                        t = float(((c1[0] - c0[0]) * d1[1] - (c1[1] - c0[1]) * d1[0]) / cross)
                        v = c0 + t * d0
                        verts.append(v if np.linalg.norm(v - p_raw) < 12.0 else p_raw)
                        continue
                if s0[2] or s1[2]:
                    sp = s0 if s0[2] else s1
                    c, d = sp[3], sp[4]
                    verts.append(c + d * float(np.dot(p_raw - c, d)))
                    continue
                verts.append(p_raw)
    verts = np.array(verts, np.float64)
    segs = []
    for k in range(m):
        p0, p1 = verts[k], verts[(k + 1) % m]
        if spans[k][2]:
            segs.append((p1[0] * scale, p1[1] * scale, p0[0] * scale, p0[1] * scale,
                         p1[0] * scale, p1[1] * scale))          # straight: controls on the ends
            continue
        seg = spans[k][6]
        # a long curved edge needs more smoothing than a short one, or it follows the pixel noise
        sig_k = float(min(max(len(seg) / 25.0, sigma), 4.0))
        pts = _smooth_closed(seg, 0.0) if len(seg) < 8 else _smooth_open(seg, sig_k)
        pts = cv2.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2), eps, False).reshape(-1, 2)
        pts = np.vstack([p0, pts[1:-1], p1]) if len(pts) > 2 else np.vstack([p0, p1])
        for q in range(len(pts) - 1):
            a0, b0 = pts[q], pts[q + 1]
            prev = pts[q - 1] if q > 0 else verts[k - 1]
            nxt = pts[q + 2] if q + 2 < len(pts) else verts[(k + 2) % m]
            t0, t1 = (b0 - prev) / 6.0, (nxt - a0) / 6.0
            L = np.linalg.norm(b0 - a0)
            for t in (t0, t1):
                nt = np.linalg.norm(t)
                if nt > 0.45 * L and nt > 0:
                    t *= 0.45 * L / nt
            c1, c2 = a0 + t0, b0 - t1
            segs.append((b0[0] * scale, b0[1] * scale, c1[0] * scale, c1[1] * scale,
                         c2[0] * scale, c2[1] * scale))
    if len(segs) < 2:
        return None
    return (verts[0][0] * scale, verts[0][1] * scale), segs


def _smooth_open(pts, sigma):
    """Gaussian smoothing of an open polyline (ends held)."""
    if len(pts) < 5 or sigma <= 0:
        return pts.astype(np.float64)
    r = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    ext = np.vstack([np.repeat(pts[:1], r, axis=0), pts, np.repeat(pts[-1:], r, axis=0)]).astype(np.float64)
    return np.stack([np.convolve(ext[:, 0], k, mode='valid'), np.convolve(ext[:, 1], k, mode='valid')], axis=1)


def _measure_outline(mask, src, lab, rim_px, k2, S, fill_lab=None):
    """The drawn rim along one shape's boundary -> {'color', 'width_px'} or None."""
    edge = (mask > 0) & ~(cv2.erode(mask, k2) > 0)
    ep = int(edge.sum())
    if ep < 12:
        return None
    covered = cv2.dilate(rim_px.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if float((edge & covered).sum()) < 0.5 * ep:
        return None
    ring = cv2.dilate(edge.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    own = ring & rim_px
    rp = src[own]
    if len(rp) < 5:
        return None
    Lr = lab[own][:, 0]
    dk = float(np.percentile(Lr, 15))
    oc = np.median(rp[Lr <= dk], axis=0)[::-1]
    wpx = float(int((Lr <= dk + 25).sum())) / float(max(ep, 1))
    if wpx < 0.28:
        return None
    if fill_lab is not None:
        oc_lab = cv2.cvtColor(np.uint8([[list(oc)[::-1]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
        if float(np.linalg.norm(oc_lab - fill_lab)) < 45:
            return None                           # just a darker edge of the same colour (JPEG ringing)
    return {'color': [int(v) for v in oc], 'width_px': round(float(min(max(wpx * 1.8, 0.7), 3.0)) * S, 2)}


def build_flat(work_dir, meta, S=1, log=print):
    """Replace meta['layers'] / meta['rules'] with the flat-colour model. Returns stats."""
    from cdr_vectorize import imread, imwrite
    from cdr_vectorize_color import dashed_strokes

    src = imread(os.path.join(work_dir, 'src.png'), cv2.IMREAD_COLOR)
    h, w = src.shape[:2]
    lab = _lab(src)
    std = flatness(lab)
    flat = std < FLAT_STD
    cen, cnt = palette(lab, flat)
    idx, dist = nearest(lab, cen)
    seedable = flat & (dist < SEED_DE)

    # ---- classes: region colours have solid areas; line colours are only ever thin ---------------------
    ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kinds = []
    for j in range(len(cen)):
        m = ((idx == j) & (dist < SEED_DE + 6)).astype(np.uint8)
        a = int(m.sum())
        core = int(cv2.erode(m, ell).sum())
        kinds.append('region' if a and core >= 0.25 * a and core >= 30 else 'line')
    region_ids = [j for j, k in enumerate(kinds) if k == 'region']
    L = cen[:, 0]
    chroma = np.hypot(cen[:, 1] - 128, cen[:, 2] - 128)
    # paper: the lightest near-neutral region colour touching the border
    border = np.zeros((h, w), bool)
    border[:2, :] = border[-2:, :] = True
    border[:, :2] = border[:, -2:] = True
    paper_id = None
    for j in sorted(region_ids, key=lambda j_: -L[j_]):
        if L[j] > 235 and chroma[j] < 8 and ((idx == j) & border & seedable).any():
            paper_id = j
            break

    # ---- text: inside each label box, what is not the box's own background -------------------------
    text = np.zeros((h, w), bool)
    for lb in meta.get('labels', []):
        bt, bb = lb.get('tight') or lb.get('box'), lb.get('box') or lb.get('tight')
        bx = [min(bt[0], bb[0]), min(bt[1], bb[1]), max(bt[2], bb[2]), max(bt[3], bb[3])]
        x0, y0, x1, y1 = [int(round(v / S)) for v in bx]
        x0, y0, x1, y1 = max(x0 - 2, 0), max(y0 - 2, 0), min(x1 + 2, w - 1), min(y1 + 2, h - 1)
        if x1 <= x0 or y1 <= y0:
            continue
        rx0, ry0, rx1, ry1 = max(x0 - 4, 0), max(y0 - 4, 0), min(x1 + 4, w - 1), min(y1 + 4, h - 1)
        ring = np.zeros((h, w), bool)
        ring[ry0:ry1 + 1, rx0:rx1 + 1] = True
        ring[y0:y1 + 1, x0:x1 + 1] = False
        ids_r, cnt_r = np.unique(idx[ring & seedable], return_counts=True)
        tot = max(int(cnt_r.sum()), 1)
        bg = [int(i) for i, c in zip(ids_r, cnt_r) if c >= 0.12 * tot and kinds[int(i)] == 'region']
        box = np.zeros((h, w), bool)
        box[y0:y1 + 1, x0:x1 + 1] = True
        # background by distance to the background colour itself, generously: the JPEG ringing round the
        # glyphs is off-colour enough to miss the palette and it linked the glyphs to pixels outside the box
        is_bg = np.zeros((h, w), bool)
        wy0, wy1, wx0, wx1 = max(y0 - 20, 0), min(y1 + 21, h), max(x0 - 20, 0), min(x1 + 21, w)
        for b_ in bg:
            is_bg[wy0:wy1, wx0:wx1] |= np.linalg.norm(lab[wy0:wy1, wx0:wx1] - cen[b_], axis=2) < 22
        # text-like: neither the background nor another region colour (a band passing the label)
        # 'other' = region colours that are in the RING too (a band running through the box). The text colour
        # itself can be a region colour (thick strokes have flat pixels) but it does not surround the box.
        passing = [int(i) for i, c in zip(ids_r, cnt_r) if c >= 0.03 * tot and int(i) not in bg
                   and kinds[int(i)] == 'region']
        other = np.isin(idx, passing) & seedable
        textish = (~is_bg & ~other).astype(np.uint8)
        win = cv2.dilate(box.astype(np.uint8), np.ones((31, 31), np.uint8))
        n_, lb_ = cv2.connectedComponents(textish & win, connectivity=8)
        grow = cv2.dilate(box.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        leaving = set(np.unique(lb_[(lb_ > 0) & ~grow]).tolist())
        for q in np.unique(lb_[box & (textish > 0)]).tolist():
            if q == 0 or q in leaving:
                continue                          # a line crossing the label runs out of it: linework
            text |= (lb_ == q) & box
        # and whatever in the box has the label's own ink colour, whatever it touches: "漏" was joined to
        # the seep symbol beside it, which runs out of the box, so the glyph passed for linework
        tc = lb.get('color')
        if tc:
            tl = cv2.cvtColor(np.uint8([[tc[::-1]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
            text |= box & (textish > 0) & (np.linalg.norm(lab - tl, axis=2) < 35)
    text = cv2.dilate(text.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

    # ---- partition: watershed from region seeds ------------------------------------------------------
    markers = np.zeros((h, w), np.int32)
    for n_, j in enumerate(region_ids):
        markers[(idx == j) & seedable & ~text] = n_ + 1
    if not markers.any():
        return {'flat': False, 'reason': 'no region seeds'}
    ws = cv2.watershed(cv2.GaussianBlur(src, (3, 3), 0), markers.copy())
    # watershed ridge (-1) and anything unset: take the neighbouring label
    for _ in range(4):
        bad = ws <= 0
        if not bad.any():
            break
        dil = cv2.dilate(np.where(bad, 0, ws).astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
        ws[bad] = dil[bad]
    ws[ws <= 0] = 1
    # Edge refinement: the flood puts a boundary somewhere in the anti-aliased band; the true edge is where
    # the colour is half-way between the two regions. A non-seed pixel that is a BLEND of two adjacent
    # regions goes to the one it is nearer on that blend - star tips (a thin red spike blended into yellow)
    # survive instead of being shaved. Pixels that are no such blend (lines, text) keep the flood's answer.
    k5 = np.ones((5, 5), np.uint8)
    nearm = [cv2.dilate((ws == n_ + 1).astype(np.uint8), k5) > 0 for n_ in range(len(region_ids))]
    free_px = (~seedable) & (~text)
    best_seg = np.full((h, w), 1e9, np.float32)
    best_lab = ws.copy()
    for a_ in range(len(region_ids)):
        for b_ in range(a_ + 1, len(region_ids)):
            both = free_px & nearm[a_] & nearm[b_]
            if not both.any():
                continue
            A, B = cen[region_ids[a_]], cen[region_ids[b_]]
            ab = B - A
            P = lab[both]
            t = np.clip(((P - A) @ ab) / max(float(ab @ ab), 1.0), 0, 1)
            seg = np.linalg.norm(P - (A + t[:, None] * ab), axis=1)
            cur = best_seg[both]
            upd = seg < cur
            ys_b, xs_b = np.nonzero(both)
            best_seg[ys_b[upd], xs_b[upd]] = seg[upd]
            best_lab[ys_b[upd], xs_b[upd]] = np.where(t[upd] < 0.5, a_ + 1, b_ + 1)
    ref = free_px & (best_seg < 10)
    ws[ref] = best_lab[ref]
    # rescue thin real elements of a region colour (arrow tails): exact-colour pixels the flood took away
    for n_, j in enumerate(region_ids):
        exact = ((idx == j) & (dist < 7) & (ws != n_ + 1) & ~text).astype(np.uint8)
        if exact.any():
            k_, lb_, st_, _ = cv2.connectedComponentsWithStats(exact, 8)
            for q in range(1, k_):
                if st_[q, 4] >= 6:
                    ws[lb_ == q] = n_ + 1
    # merge tiny components into their dominant neighbour
    merged = 0
    for n_ in range(1, len(region_ids) + 1):
        m = (ws == n_).astype(np.uint8)
        k_, lb_, st_, _ = cv2.connectedComponentsWithStats(m, 8)
        for q in range(1, k_):
            if st_[q, 4] >= REGION_MIN_PX:
                continue
            x, y, bw, bh, _a = st_[q]
            x0, y0, x1, y1 = max(x - 1, 0), max(y - 1, 0), min(x + bw + 1, w), min(y + bh + 1, h)
            comp = lb_[y0:y1, x0:x1] == q
            ring = (cv2.dilate(comp.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~comp
            nb = ws[y0:y1, x0:x1][ring]
            nb = nb[nb != n_]
            if len(nb):
                ws[y0:y1, x0:x1][comp] = np.bincount(nb).argmax()
                merged += 1

    # ---- lines: what the partition does not explain -------------------------------------------------
    # A pixel is linework when it matches no region colour nearby and is not a blend of two neighbouring
    # regions either (the anti-aliased edge between red and tan is 40 units from both, but it lies on the
    # red-tan segment). Line colours are then clustered from those pixels - they never come from the flat
    # palette, because a 2 px line has no flat pixels.
    reg_c = [cen[j] for j in region_ids]
    near = []
    k5 = np.ones((5, 5), np.uint8)
    for n_ in range(len(region_ids)):
        near.append(cv2.dilate((ws == n_ + 1).astype(np.uint8), k5) > 0)
    dmin = np.full((h, w), 1e9, np.float32)
    for n_, c in enumerate(reg_c):
        d = np.linalg.norm(lab - c, axis=2)
        dmin = np.where(near[n_], np.minimum(dmin, d), dmin)
    cand = (dmin > 16) & ~text
    if cand.any():
        for a_ in range(len(reg_c)):
            for b_ in range(a_ + 1, len(reg_c)):
                both = cand & near[a_] & near[b_]
                if not both.any():
                    continue
                A, B = reg_c[a_], reg_c[b_]
                ab = B - A
                P = lab[both]
                t = np.clip(((P - A) @ ab) / max(float(ab @ ab), 1.0), 0, 1)
                seg = np.linalg.norm(P - (A + t[:, None] * ab), axis=1)
                dd = dmin[both]
                dmin[both] = np.minimum(dd, seg)
        cand = (dmin > 16) & ~text
    line_mask = cand.astype(np.uint8)
    k_, lb_, st_, _ = cv2.connectedComponentsWithStats(line_mask, 8)
    for q in range(1, k_):
        if st_[q, 4] < 8:
            line_mask[lb_ == q] = 0
    # Region edges drawn in the source (a thin grey rim round every stratum, a legend swatch's border)
    # are OUTLINES of the regions, not free lines: as strokes they broke into thousands of pieces. Line
    # pixels hugging a partition boundary are set aside; each region whose boundary is mostly rimmed gets
    # an outline in that colour below.
    bnd = np.zeros((h, w), bool)
    bnd[:, 1:] |= ws[:, 1:] != ws[:, :-1]
    bnd[1:, :] |= ws[1:, :] != ws[:-1, :]
    rim_band = cv2.dilate(bnd.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0   # a 3 px sea-floor rim fits
    rim_px = (line_mask > 0) & rim_band
    # per pixel, not per component: every fault crosses some boundary, so as components the rims and the
    # faults were one connected tangle. A fault loses only its few crossing pixels here; the dash chainer
    # bridges such gaps.
    stroke_px = (line_mask > 0) & ~rim_band

    strokes_layers = []
    if stroke_px.any():
        # group line components by the colour of their core (AA strength varies along one line)
        n_s, lb_s, st_s, _ = cv2.connectedComponentsWithStats(stroke_px.astype(np.uint8), 8)
        groups = []                               # (core lab, [component ids])
        for q in range(1, n_s):
            if st_s[q, 4] < 8:
                continue
            x, y, bw, bh, _a = st_s[q]
            comp = lb_s[y:y + bh, x:x + bw] == q
            Pl = lab[y:y + bh, x:x + bw][comp]
            core = Pl[Pl[:, 0] <= np.percentile(Pl[:, 0], 40)].mean(axis=0)
            cc_ = float(np.hypot(core[1] - 128, core[2] - 128))
            for g in groups:
                if np.linalg.norm(g[0] - core) < 25 and abs(float(np.hypot(g[0][1] - 128, g[0][2] - 128)) - cc_) < 10:
                    g[1].append(q)
                    break
            else:
                groups.append([core, [q]])
        # the same ink at different anti-aliasing strength forms several close groups: merge them first, so
        # a vote is between genuinely different inks (19 near-duplicate groups scattered the arrow's votes)
        merged_g = []
        for core, ids in sorted(groups, key=lambda g: -len(g[1])):
            for mg in merged_g:
                if np.linalg.norm(mg[0] - core) < 30:
                    mg[1].extend(ids)
                    break
            else:
                merged_g.append([core, list(ids)])
        groups = merged_g
        # a component that mixes two clearly different inks (an arrow touching a fault) is split per pixel:
        # each pixel goes to the ink whose blend with the local region colour explains it best
        bg_lab = np.zeros((h, w, 3), np.float32)
        for n_, j in enumerate(region_ids):
            bg_lab[ws == n_ + 1] = cen[j]
        cores = [g[0] for g in groups]
        gid_of = {q: gi for gi, g in enumerate(groups) for q in g[1]}
        pix_group = np.full((h, w), -1, np.int32)
        if len(groups) > 1:
            ys_, xs_ = np.nonzero(np.isin(lb_s, list(gid_of)))
            P = lab[ys_, xs_]
            Bk = bg_lab[ys_, xs_]
            dg = []
            for core in cores:
                ab = Bk - core
                t = np.clip(np.einsum('ij,ij->i', P - core, ab) / np.maximum(np.einsum('ij,ij->i', ab, ab), 1.0),
                            0, 0.85)
                dg.append(np.linalg.norm(P - (core + t[:, None] * ab), axis=1))
            dgs = np.stack(dg, axis=1)
            vote = np.argmin(dgs, axis=1)
            comp = lb_s[ys_, xs_]
            final = np.array([gid_of[int(c_)] for c_ in comp])
            for q in np.unique(comp):
                sel_q = comp == q
                v = np.bincount(vote[sel_q], minlength=len(groups))
                # split only when a second ink holds a real share and is far from the first
                main = gid_of[int(q)]
                far = [g_ for g_ in range(len(groups)) if g_ != main and v[g_] >= 30
                       and np.linalg.norm(cores[g_] - cores[main]) > 45]
                if far:
                    fq = final[sel_q]
                    vq = vote[sel_q]
                    dq = dgs[sel_q]
                    for g_ in far:
                        # only where the other ink is CLEARLY better: an anti-aliased edge of a navy dash is
                        # about as well explained by a light grey, and those ties chopped the faults up
                        take = (vq == g_) & (dq[:, g_] + 8 < dq[:, main])
                        fq[take] = g_
                    final[sel_q] = fq
            pix_group[ys_, xs_] = final
        else:
            pix_group[np.isin(lb_s, list(gid_of))] = 0
        for gi, (core, ids) in enumerate(groups):
            m = (pix_group == gi).astype(np.uint8)
            if int(m.sum()) < 30:
                continue
            Pb = src[m > 0]
            Lc = lab[m > 0][:, 0]
            rgb = [int(v) for v in np.median(Pb[Lc <= np.percentile(Lc, 40)], axis=0)[::-1]]
            # dashes are judged on the CORE of the line: the anti-aliased ends bridge a 3 px gap and the
            # dashes of a fault fused into one solid line
            m_core = (m.astype(bool) & (np.linalg.norm(lab - core, axis=2) < 30)).astype(np.uint8)
            dashed = dashed_strokes(m_core, rgb) or dashed_strokes(m, rgb)
            rest = m
            if dashed:
                st2, _rest = dashed
                cover = np.zeros((h, w), np.uint8)
                for pl in st2['polylines']:
                    pts = np.array(pl, np.float32).reshape(-1, 2).astype(np.int32)
                    cv2.polylines(cover, [pts], False, 1, thickness=int(max(3, 2 * st2['width_px'] + 3)))
                rest = (m.astype(bool) & ~(cover > 0)).astype(np.uint8)
                st2 = dict(st2)
                st2['polylines'] = [[v * S for v in pl] for pl in st2['polylines']]
                st2['width_px'] = st2['width_px'] * S
                st2['dash'] = [v * S for v in st2['dash']]
                strokes_layers.append({'kind': 'color', 'color': rgb, 'area': int(m.sum()), 'strokes': st2})
            if int(rest.sum()) >= 20:
                polys, width = _chain_centerlines(rest)
                # pieces shorter than 6 px are crumbs of a rim or of a crossing, not a line of the drawing
                polys = [[v * S for v in pl] for pl in polys if len(pl) >= 4 and
                         sum(math.dist(pl[q:q + 2], pl[q + 2:q + 4]) for q in range(0, len(pl) - 2, 2)) >= 6]
                if polys:
                    strokes_layers.append({'kind': 'color', 'color': rgb, 'area': int(rest.sum()),
                                           'strokes': {'polylines': polys, 'width_px': round(width * S, 2),
                                                       'color': rgb}})
    # A rim is a line someone DREW along the boundary, so it must be darker than both regions it separates.
    # Without that test the anti-aliased edge of a red star on grey counted as a rim and every star got a
    # dark red outline, while the legend's real 2 px black border came out as one global thin grey line.
    Lch = lab[..., 0]
    reg_L = np.full((h, w), 255.0, np.float32)
    for n_, j in enumerate(region_ids):
        reg_L[ws == n_ + 1] = cen[j][0]
    near_L = cv2.erode(reg_L, np.ones((7, 7), np.uint8))          # the darker of the regions around it
    rim_px &= Lch < near_L - 8

    # ---- sub-pixel edges: the partition refined at 2x --------------------------------------------------
    # At 1x a boundary can only sit between pixels, so a star tip one pixel wide is either all red or gone
    # and every band edge is a staircase that smoothing then rounds off. On the bicubic 2x image the
    # anti-aliased edge is resolved: each boundary pixel goes to the adjacent region it is nearer to on
    # their blend, which puts the edge where the drawing had it.
    up_lab = _lab(cv2.resize(src, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC))
    ws2 = np.repeat(np.repeat(ws, 2, axis=0), 2, axis=1)
    b2 = np.zeros(ws2.shape, bool)
    b2[:, 1:] |= ws2[:, 1:] != ws2[:, :-1]
    b2[1:, :] |= ws2[1:, :] != ws2[:-1, :]
    band2 = cv2.dilate(b2.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    hold = np.repeat(np.repeat(text | (line_mask > 0), 2, axis=0), 2, axis=1)
    free2 = band2 & ~hold
    k5b = np.ones((5, 5), np.uint8)
    present = [n_ for n_ in range(len(region_ids)) if (ws2[band2] == n_ + 1).any()]
    near2 = {n_: cv2.dilate((ws2 == n_ + 1).astype(np.uint8), k5b) > 0 for n_ in present}
    bseg = np.full(ws2.shape, 1e9, np.float32)
    blab = ws2.copy()
    for ia, a_ in enumerate(present):
        for b_ in present[ia + 1:]:
            both = free2 & near2[a_] & near2[b_]
            if not both.any():
                continue
            A, B = cen[region_ids[a_]], cen[region_ids[b_]]
            ab = B - A
            P = up_lab[both]
            t = np.clip(((P - A) @ ab) / max(float(ab @ ab), 1.0), 0, 1)
            seg = np.linalg.norm(P - (A + t[:, None] * ab), axis=1)
            yy, xx = np.nonzero(both)
            upd = seg < bseg[yy, xx]
            bseg[yy[upd], xx[upd]] = seg[upd]
            blab[yy[upd], xx[upd]] = np.where(t[upd] < 0.5, a_ + 1, b_ + 1)
    sel2 = free2 & (bseg < 12)
    ws2[sel2] = blab[sel2]

    # ---- shapes: one per connected region, outer-first ----------------------------------------------
    shapes = []
    k2 = np.ones((3, 3), np.uint8)
    for n_, j in enumerate(region_ids):
        m = (ws == n_ + 1).astype(np.uint8)
        if not m.any():
            continue
        px = src[m > 0]
        rgb = [int(v) for v in np.median(px[dist[m > 0] < SEED_DE], axis=0)[::-1]] if (dist[m > 0] < SEED_DE).any() \
            else [int(v) for v in np.median(px, axis=0)[::-1]]
        # the refined 2x partition; grown by one 2x pixel so neighbours overlap instead of leaving a seam
        up = (ws2 == n_ + 1).astype(np.uint8)
        up = cv2.morphologyEx(up, cv2.MORPH_OPEN, k2) | (up & cv2.erode(up, np.ones((2, 2), np.uint8)))
        up = cv2.dilate(up, k2)
        # with its HOLES: painting outer-first and letting smaller regions cover the holes failed when a
        # region reached round others through a thin strip (the carbonate band enclosed the whole frame,
        # was painted first and vanished under the sediment). Real holes make stacking order irrelevant.
        cnts, hier = cv2.findContours(up, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hier = hier[0] if hier is not None else []
        holes_of = {}
        for ci, hq in enumerate(hier):
            if hq[3] >= 0:
                holes_of.setdefault(int(hq[3]), []).append(ci)
        # outline: per SHAPE, not per colour. Measured over a whole colour, the legend swatch's 2 px black
        # frame was averaged with the 1 px rim of the band that shares its colour and came out at 0.37 px.
        n_cc, cc = cv2.connectedComponents(m, connectivity=8)
        outline_of = {}
        for ci_ in range(1, n_cc):
            outline_of[ci_] = _measure_outline((cc == ci_).astype(np.uint8), src, lab, rim_px, k2, S,
                                               fill_lab=cen[j])
        edge = (m > 0) & ~(cv2.erode(m, k2) > 0)
        ring = cv2.dilate(edge.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        ep = int(edge.sum())
        outline = None
        if ep:
            covered = cv2.dilate(rim_px.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            if float((edge & covered).sum()) >= 0.5 * ep:
                own = ring & rim_px                  # this region's own rim: colour AND width from it
                rp = src[own]
                Lr = lab[own][:, 0]
                if len(rp) >= 5:
                    # The drawn line is the DARKEST part of the rim; the rest is its anti-aliasing. Taking
                    # the median over all of it diluted a 2 px near-black legend border to a thin light
                    # brown. Colour from the darkest 15%, width from the pixels that share that colour.
                    dk = float(np.percentile(Lr, 15))
                    oc = np.median(rp[Lr <= dk], axis=0)[::-1]
                    core_px = int((Lr <= dk + 25).sum())
                    wpx = float(core_px) / float(max(ep, 1))
                    # a drawn line is at least about a pixel wide; the dark JPEG ringing around the
                    # saturated red stars measured 0.74 px and would have given every star an outline
                    # wpx counts only the dark core along one side of the boundary; measured against the
                    # source this drawing's 2 px borders came out at 0.79, so the drawn width is ~2.4x it.
                    # The JPEG ringing round the red stars measures 0.13 and stays outline-free.
                    if wpx >= 0.28:
                        outline = {'color': [int(v) for v in oc],
                                   'width_px': round(float(min(max(wpx * 2.4, 0.7), 3.5)) * S, 2)}
        for ci, c in enumerate(cnts):
            if len(hier) and hier[ci][3] >= 0:
                continue                          # a hole: emitted with its outer contour
            filled_area = float(cv2.contourArea(c)) / 4.0
            if filled_area < 2:
                continue
            sig = 0.7 if filled_area < 1500 else (1.2 if filled_area < 8000 else 1.6)
            bz = contour_to_bezier(c, scale=S / 2.0, eps=0.6 if filled_area < 1500 else 1.0, sigma=sig)
            if bz is None:
                continue
            paths = [{'start': [round(v, 2) for v in bz[0]], 'segs': [[round(v, 2) for v in q] for q in bz[1]]}]
            for hi in holes_of.get(ci, []):
                if float(cv2.contourArea(cnts[hi])) / 4.0 < 3:
                    continue
                hz = contour_to_bezier(cnts[hi], scale=S / 2.0, eps=0.8, sigma=sig)
                if hz is not None:
                    paths.append({'start': [round(v, 2) for v in hz[0]],
                                  'segs': [[round(v, 2) for v in q] for q in hz[1]]})
            cx, cy = c.reshape(-1, 2).mean(axis=0) / 2.0
            ci_ = int(cc[int(min(max(cy, 0), h - 1)), int(min(max(cx, 0), w - 1))])
            if ci_ == 0:                          # centroid outside the shape: take any of its pixels
                pts_ = c.reshape(-1, 2)[0] / 2.0
                ci_ = int(cc[int(min(max(pts_[1], 0), h - 1)), int(min(max(pts_[0], 0), w - 1))])
            shapes.append({'color': rgb, 'paths': paths, 'filled': filled_area,
                           'paper': j == paper_id, 'outline': outline_of.get(ci_, outline)})
    shapes.sort(key=lambda s: -s['filled'])
    # paper-coloured shapes that sit on nothing are the page itself: not drawn
    drawn = [s for s in shapes if not (s['paper'] and s['filled'] > 0.3 * h * w)]

    # Rim left over: a drawn line along a boundary is reproduced as a region OUTLINE, which only works when
    # it goes all the way round a shape and whose width has to be inferred. Wherever no outline covers it,
    # draw the rim as what it is - a vector line, with the width measured straight off the pixels. That is
    # what the element check kept reporting as 2 px borders drawn at 1.5 px or missing at one edge.
    outlined_px = np.zeros((h, w), np.uint8)
    for s_ in drawn:
        ol = s_.get('outline')
        if not ol:
            continue
        for p_ in s_['paths']:
            pts = [p_['start']] + [q[:2] for q in p_['segs']]
            arr = np.round(np.array(pts, np.float64) / max(S, 1)).astype(np.int32)
            if len(arr) >= 2:
                cv2.polylines(outlined_px, [arr], True, 1,
                              max(5, int(round(float(ol['width_px']) / max(S, 1))) + 5))
    rim_left = (rim_px & (outlined_px == 0)).astype(np.uint8)
    rim_left = cv2.morphologyEx(rim_left, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    k_, lb_, st_, _ = cv2.connectedComponentsWithStats(rim_left, 8)
    for q in range(1, k_):
        if st_[q, 4] < 12 or max(st_[q, 2], st_[q, 3]) < 8:
            rim_left[lb_ == q] = 0                 # crumbs, not a line
    if int(rim_left.sum()) >= 60:
        polys, width = _chain_centerlines(rim_left)
        polys = [[v * S for v in pl] for pl in polys if len(pl) >= 4 and
                 sum(math.dist(pl[q:q + 2], pl[q + 2:q + 4]) for q in range(0, len(pl) - 2, 2)) >= 6]
        if polys:
            rp = src[rim_left > 0]
            Lr = lab[rim_left > 0][:, 0]
            rgb_rim = [int(v) for v in np.median(rp[Lr <= np.percentile(Lr, 40)], axis=0)[::-1]]
            strokes_layers.append({'kind': 'color', 'color': rgb_rim, 'area': int(rim_left.sum()),
                                   'strokes': {'polylines': polys, 'width_px': round(width * S, 2),
                                               'color': rgb_rim}})

    os.makedirs(os.path.join(work_dir, 'layers'), exist_ok=True)
    vis = np.zeros((h, w, 3), np.uint8)
    for n_, j in enumerate(region_ids):
        vis[ws == n_ + 1] = cv2.cvtColor(np.uint8([[np.clip(cen[j], 0, 255)]]), cv2.COLOR_LAB2BGR)[0, 0]
    imwrite(os.path.join(work_dir, 'layers', 'flat_partition.png'), vis)
    imwrite(os.path.join(work_dir, 'layers', 'flat_text.png'), np.where(text, 0, 255).astype(np.uint8))
    layers = [{'kind': 'regions', 'file': 'layers/flat_partition.png', 'area': int(h * w),
               'color': [0, 0, 0], 'shapes': [{'color': s['color'], 'paths': s['paths'],
                                               **({'outline': s['outline']} if s['outline'] else {})}
                                              for s in drawn]}]
    # thin outlines first, dashed faults next, dark ink on top
    strokes_layers.sort(key=lambda l: (bool(l['strokes'].get('dash')), -sum(l['color'])))
    for i, l in enumerate(strokes_layers):
        l['file'] = 'layers/flat_partition.png'
        layers.append(l)
    meta['layers'] = layers
    meta['rules'] = None
    meta['flat'] = {'regions': len(drawn), 'region_colours': len(region_ids), 'line_layers': len(strokes_layers),
                    'merged_specks': merged, 'text_px': int(text.sum()),
                    'strokes': sum(len(l['strokes']['polylines']) for l in strokes_layers)}
    log('flat: %s' % meta['flat'])
    return meta['flat']
