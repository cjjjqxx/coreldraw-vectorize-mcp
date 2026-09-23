"""Colour layering for the vectorize pipeline.

A colour drawing is split into flat layers instead of letting PowerTRACE quantise colour itself:
the image is reduced to a small palette, each palette colour becomes one black-on-white mask that
is traced separately and filled with that colour, and the dark neutral "ink" (lines + text) becomes
the top layer. This keeps region edges clean and keeps the linework from being broken up by colour.

CLI: python cdr_vectorize_color.py layers <work_dir> <labels_confirmed.json> [colors]
"""
from __future__ import annotations

import json
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdr_vectorize import (S, MEASURED_TEXT, QUAD_WIPE, COLOUR_WIPE, DASH_PROTECT,  # noqa: E402
                           annotate_placement, erase_text, expand_to_text, imread, imwrite,
                           _line_mask, _rule_mask, _heal_crossings,
                           _split_box, _glyph_band, measure_text_h, is_bold, blur_factor, detect_latin_font)

WHITE_MIN = 232          # a pixel whose channels are all above this counts as paper
MERGE_LAB_DIST = 11.0    # palette entries closer than this in Lab are merged


def _lab(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)


def extract_palette(bgr, k=6, seed=0):
    """k-means over non-paper pixels in Lab; close centres merged, tiny ones dropped."""
    small = cv2.resize(bgr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    lab = _lab(small).reshape(-1, 3).astype(np.float32)
    keep = small.reshape(-1, 3).min(axis=1) < WHITE_MIN
    data = lab[keep]
    if len(data) < k:
        return np.array([[0, 128, 128]], np.float32)
    if len(data) > 120000:
        rng = np.random.default_rng(seed)
        data = data[rng.choice(len(data), 120000, replace=False)]
    # weight chroma above lightness, otherwise small saturated regions get swallowed by greys
    wt = np.array([0.5, 1.6, 1.6], np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    _, _, centers_w = cv2.kmeans(data * wt, k, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    centers = centers_w / wt
    merged = []
    for c in sorted(centers, key=lambda c: c[0]):
        if all(np.linalg.norm(c - m) > MERGE_LAB_DIST for m in merged):
            merged.append(c)
    return np.array(merged, np.float32)


def assign_palette(bgr, centers):
    """Per-pixel nearest palette entry; paper stays unassigned (-1)."""
    lab = _lab(bgr).astype(np.float32)
    h, w = lab.shape[:2]
    flat = lab.reshape(-1, 3)
    d = np.stack([np.linalg.norm(flat - c, axis=1) for c in centers], axis=1)
    idx = d.argmin(axis=1).astype(np.int16)
    paper = bgr.reshape(-1, 3).min(axis=1) >= WHITE_MIN
    idx[paper] = -1
    return idx.reshape(h, w)


def classify(center_lab):
    """ink = dark neutral (black linework + text), grid = light neutral (graticule / thin grey
    rules, merged into one layer), everything else is a colour region."""
    L, a, b = float(center_lab[0]), float(center_lab[1]) - 128, float(center_lab[2]) - 128
    chroma = (a * a + b * b) ** 0.5
    if chroma < 22 and L < 130:
        return 'ink'
    if chroma < 16 and L >= 130:
        return 'grid'
    return 'color'


def lab_to_bgr(c):
    px = np.uint8([[[np.clip(c[0], 0, 255), np.clip(c[1], 0, 255), np.clip(c[2], 0, 255)]]])
    return cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0]


def measure_glyph_h(ink_nh, box):
    """Median height of the glyph-sized components inside the box. A fault line crossing the label
    would dominate a simple min/max band, so use the median of plausible glyph pieces."""
    x0, y0, x1, y1 = box
    sub = ink_nh[y0:y1 + 1, x0:x1 + 1]
    n, _, st, _ = cv2.connectedComponentsWithStats(sub, 8)
    box_h = max(y1 - y0, 1)
    hs = [int(st[i][3]) for i in range(1, n)
          if st[i][4] >= 4 and 0.25 * box_h <= st[i][3] <= box_h]
    if hs:
        return int(np.median(hs))
    return int(_glyph_band(ink_nh, box)[0])


def thicken_ink(bgr, ink_mask):
    """Hysteresis ink mask. Thin source rules (~1 px) come out of SR as a 1-2 px dark core whose
    anti-aliased flanks k-means hands to the grey/grid cluster; PowerTRACE silently drops such
    hairlines (whole fault-line segments and roman numerals vanished). Grow the dark neutral core by
    one ring into the neutral mid-grey flank so every stroke is >= 3 px wide before tracing. Only one
    ring: a full flood fill would leak along the graticule at every crossing."""
    lab = _lab(bgr).astype(np.int16)
    L = lab[..., 0]
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    core = ink_mask.astype(bool) | ((L < 90) & (chroma < 20))
    ring = cv2.dilate(core.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    flank = ring & (L < 232) & (chroma < 25)
    return (core | flank).astype(np.uint8)


def drop_region_rims(ink_mask, color_px):
    """The darker anti-aliased rim of a filled colour patch lands in the dark-neutral ink cluster and
    traces as black crumbs around every patch. Drop small ink pieces that mostly hug colour."""
    near = cv2.dilate(color_px.astype(np.uint8), np.ones((2 * S + 1, 2 * S + 1), np.uint8)) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink_mask.astype(np.uint8), 8)
    out = ink_mask.copy()
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if max(bw, bh) > 12 * S:
            continue
        comp = lab[y:y + bh, x:x + bw] == i
        if near[y:y + bh, x:x + bw][comp].mean() > 0.5:
            out[y:y + bh, x:x + bw][comp] = 0
    return out


def drop_patch_outlines(ink_mask, color_px, src_gray=None, text_zone=None):
    """Filled patches carry a 1-2 px darker outline that lands in the ink layer as long thin rims
    (drop_region_rims only catches short crumbs). Remove thin ink (no 5 px wide core) whose
    connected piece mostly hugs a colour patch; fault lines are thick and kept, and a small map
    symbol that merely touches a patch is only partly near it, so it survives.

    On a map that is mostly filled regions, though, nearly every thin line hugs a patch, and "only
    partly near it" does not save a symbol drawn INSIDE one. Whole map furniture went this way:
    the inset's study-area rectangle, the place-name squares, dashed region boundaries, even parts
    of the lettering - 127k px deleted against 122k kept. What the test cannot see is that a drawn
    line is dark in the SOURCE while an SR rim is a mid-grey transition, so the decision is made per
    pixel on the source, with the same "genuinely dark" bar split_faint_ink uses (L < 125). Flat
    patch interiors are unaffected: their rims stay above it and are still dropped."""
    ink = (ink_mask > 0).astype(np.uint8)
    keep = None if src_gray is None else (src_gray < 125)
    zone = None if text_zone is None else (text_zone > 0)
    reach = None
    if keep is not None and zone is not None:
        # whether each WHOLE ink structure shows anywhere outside the label boxes (judged on the full
        # ink, not on the thin pieces: a frame's corners are thick and split its sides into pieces
        # that each sit entirely inside a dense inset's boxes)
        _ni, ink_cc = cv2.connectedComponents(ink, connectivity=8)
        reach = np.zeros(_ni, bool)
        reach[np.unique(ink_cc[~zone])] = True
        reach[0] = False
    disk5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thick = cv2.dilate(cv2.morphologyEx(ink, cv2.MORPH_OPEN, disk5), disk5)
    thin = ink & (1 - thick)
    near = cv2.dilate(color_px.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * S + 3, 2 * S + 3))) > 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(thin, 8)
    out = ink.copy()
    for i in range(1, n):
        x, y, bw, bh, _ = st[i]
        comp = lab[y:y + bh, x:x + bw] == i
        if near[y:y + bh, x:x + bw][comp].mean() > 0.6:
            if keep is not None:
                k_ = keep[y:y + bh, x:x + bw]
                # Inside a label box dark source pixels are usually the lettering itself: erase_text
                # keeps long straight strokes as "lines" (the bars of 量 and 型), and this test is what
                # cleared those leftovers - protecting them drew ghost strokes under every CJK label
                # of a filled flowchart. A drawn line crossing the box (the inset rectangle behind
                # "Depression") runs on out of it; a leftover stroke stays inside. So inside a box
                # only pieces that reach beyond it keep their protection.
                if reach is not None and not reach[ink_cc[y:y + bh, x:x + bw][comp]].any():
                    k_ = np.zeros_like(k_)
                comp = comp & ~k_
            out[y:y + bh, x:x + bw][comp] = 0
    return out


def drop_specks(ink_mask, max_dim=2 * S):
    """Isolated dots of a few pixels (JPEG/SR noise on the graticule) trace as black specks."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((ink_mask > 0).astype(np.uint8), 8)
    out = (ink_mask > 0).astype(np.uint8)
    small = [i for i in range(1, n) if max(st[i][2], st[i][3]) <= max_dim]
    if small:
        out[np.isin(lab, small)] = 0
    return out


def split_faint_ink(ink_mask, cmasks, centers, src_bgr):
    """SR turns faint grey marks (grey roman numerals, light unlabeled glyphs, patch outlines) and
    JPEG-faded small colour patches into solid black, so the ink layer drew them as black blobs.
    Judge each small ink piece on the SOURCE instead: dark there -> stays black ink; light there ->
    a colour layer if it still leans to that layer's hue, else a separate grey ink layer painted
    with the source grey. Returns (dark_ink, faint_mask, faint_rgb, moved_to_colour)."""
    h2, w2 = ink_mask.shape
    src_up = cv2.resize(src_bgr, (w2, h2), interpolation=cv2.INTER_NEAREST)
    lab = cv2.cvtColor(src_up, cv2.COLOR_BGR2LAB).astype(np.float32)
    hues = {i: np.arctan2(float(centers[i][2]) - 128, float(centers[i][1]) - 128) for i in cmasks}
    ink = (ink_mask > 0).astype(np.uint8)
    faint = np.zeros_like(ink)
    faint_px = []
    moved = 0
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    for j in range(1, n):
        x, y, w, h, area = st[j]
        if max(w, h) > 40 * S:
            continue                          # line networks / frames stay black
        comp = lbl[y:y + h, x:x + w] == j
        if dist[y:y + h, x:x + w][comp].max() < 1.25 * S:
            # thin strokes (roman numerals, glyphs) are diluted by downscaling/blur: a black 1 px stroke
            # reads light grey in a small source, so its source lightness says nothing - keep black
            continue
        px = lab[y:y + h, x:x + w][comp]
        dark = px[np.argsort(px[:, 0])[:max(1, len(px) * 3 // 10)]]     # darkest 30%: skip anti-aliasing
        L = float(np.median(dark[:, 0]))
        if L < 125:
            continue                          # genuinely dark in the source
        ab = dark[:, 1:].mean(axis=0) - 128
        chroma = float(np.hypot(ab[0], ab[1]))
        if cmasks and chroma >= 2.5:
            hue = np.arctan2(ab[1], ab[0])
            best = min(cmasks, key=lambda i: abs(np.angle(np.exp(1j * (hue - hues[i])))))
            # JPEG-faded patches keep only chroma 3-8 but their hue is within a few degrees of their
            # layer; neutral grey pieces measure chroma 0-1.5. Weak colour needs a tight hue match.
            tol = 35 if chroma >= 10 else 12
            if abs(np.angle(np.exp(1j * (hue - hues[best])))) <= np.radians(tol):
                cmasks[best][y:y + h, x:x + w][comp] = 1
                ink[y:y + h, x:x + w][comp] = 0
                moved += 1
                continue
        faint[y:y + h, x:x + w][comp] = 1
        ink[y:y + h, x:x + w][comp] = 0
        faint_px.append(src_up[y:y + h, x:x + w][comp][np.argsort(px[:, 0])[:max(1, len(px) * 3 // 10)]])
    rgb = None
    if faint_px:
        g = float(np.median(np.concatenate(faint_px).mean(axis=1)))
        rgb = [int(g)] * 3
    return ink, faint, rgb, moved


def patch_outline(src_bgr, mask, fill_rgb, drawn=None):
    """Filled map patches carry a thin darker outline (dark olive around green, brown around orange).
    It was removed as ink noise and never drawn, so patches looked smaller and small ones - mostly
    outline - nearly vanished. Sample the source on a band across the patch boundary: if its darker
    pixels are clearly darker than the fill, return that colour for a vector outline, else None."""
    h, w = src_bgr.shape[:2]
    m1 = cv2.resize((mask > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    k3 = np.ones((3, 3), np.uint8)
    band = (cv2.dilate(m1, k3) > 0) & ~(cv2.erode(m1, k3) > 0)
    px = src_bgr[band]
    if len(px) < 30:
        return None
    lab = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    dark = px[np.argsort(lab[:, 0])[:max(1, len(px) // 4)]]
    rim = np.median(dark, axis=0)                                            # BGR
    fill_lab = cv2.cvtColor(np.uint8([[fill_rgb[::-1]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    rim_lab = cv2.cvtColor(np.uint8([[rim]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    if fill_lab[0] - rim_lab[0] < 12:
        return None
    if drawn is not None:
        # The darkest quarter of the band is dark on any map whose patch borders are DRAWN - dashed
        # boundaries, fault lines - and the outline then redrew every one of them as a solid dark
        # contour (the geology map's dashed borders came out solid, its faults edged in brown). A
        # patch outline is a line of its own: judge it only where no other layer already draws the
        # border, and require it to be really there, a thin dark stroke along most of that stretch.
        d1 = cv2.resize(drawn.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        d1 = cv2.dilate(d1, np.ones((7, 7), np.uint8)) > 0
        edge = (m1 > 0) & ~(cv2.erode(m1, k3) > 0)
        edge[:2, :] = edge[-2:, :] = False
        edge[:, :2] = edge[:, -2:] = False
        free = edge & ~d1
        if free.sum() < 0.15 * max(int(edge.sum()), 1):
            return None
        L = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB)[..., 0]
        thin = (cv2.morphologyEx(L, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(np.int16)
                - L.astype(np.int16)) > 25
        near = cv2.dilate(thin.astype(np.uint8), k3) > 0
        if float(near[free].mean()) < 0.7:
            return None
    return [int(rim[2]), int(rim[1]), int(rim[0])]


def interior_color(bgr, mask):
    """Fill colour of a layer = median of its interior pixels. The k-means centre is pulled toward
    the darker patch outline (25-16 Ma peach came out brown)."""
    m = (mask > 0).astype(np.uint8)
    core = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * S + 1, 2 * S + 1)))
    sel = core > 0 if core.sum() >= 20 else m > 0
    med = np.median(bgr[sel], axis=0)
    return [int(med[2]), int(med[1]), int(med[0])]       # RGB


def rebuild_rules(grid_mask, max_tilt_deg=15.0, bridge=None, polylines=None):
    """Graticule / grid rules arrive from SR ~1 px wide and broken (labels, ink halos, colour patches,
    faint JPEG patches sit on them). Map graticules are also slightly tilted and curved, so the old
    exact horizontal/vertical opening cut them into short pieces that were then dropped (grid recall
    ~40%). Detect straight segments with a probabilistic Hough transform (near-horizontal and
    near-vertical only), chain segments that lie on the same rule, bridge gaps up to `bridge` px by
    joining consecutive endpoints (follows gentle curvature), draw 3 px wide so PowerTRACE keeps them."""
    m = (grid_mask > 0).astype(np.uint8)
    bridge = bridge if bridge is not None else 40 * S
    segs = cv2.HoughLinesP(m * 255, 1, np.pi / 360, threshold=6 * S, minLineLength=6 * S, maxLineGap=2 * S)
    out = np.zeros_like(m)
    if segs is None:
        return out
    groups = {'h': [], 'v': []}
    for x1, y1, x2, y2 in segs.reshape(-1, 4):
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        ang = (ang + 180) % 180
        if min(ang, 180 - ang) <= max_tilt_deg:
            if x2 < x1:
                x1, y1, x2, y2 = x2, y2, x1, y1
            groups['h'].append((float(x1), float(y1), float(x2), float(y2)))
        elif abs(ang - 90) <= max_tilt_deg:
            if y2 < y1:
                x1, y1, x2, y2 = x2, y2, x1, y1
            groups['v'].append((float(x1), float(y1), float(x2), float(y2)))
    for orient, lst in groups.items():
        # work in (u along the rule, w across it)
        pts = [((a, b, c, d) if orient == 'h' else (b, a, d, c)) for a, b, c, d in lst]
        pts.sort(key=lambda p: p[0])
        chains = []                                      # each chain: list of segments (u0, w0, u1, w1)
        for s in pts:
            u0, w0, u1, w1 = s
            slope = (w1 - w0) / max(u1 - u0, 1.0)
            best, best_d = None, None
            for ch in chains:
                cu0, cw0, cu1, cw1 = ch[-1]
                if u0 > cu1 + bridge or u0 < ch[0][0] - 2 * S:
                    continue
                cs = (cw1 - cw0) / max(cu1 - cu0, 1.0)
                if abs(np.degrees(np.arctan(slope) - np.arctan(cs))) > 4:
                    continue
                pred = cw1 + cs * (u0 - cu1)                 # extend the chain's last segment
                d = abs(pred - w0)
                if d <= 1.5 * S and (best_d is None or d < best_d):
                    best, best_d = ch, d
            if best is None:
                chains.append([s])
            elif u1 > best[-1][2]:
                best.append(s)
        for ch in chains:
            span = ch[-1][2] - ch[0][0]
            if span < 12 * S:
                continue                                     # short isolated strokes are not rules
            poly = []
            for u0, w0, u1, w1 in ch:
                poly += [(u0, w0), (u1, w1)]
            arr = np.array([(p if orient == 'h' else (p[1], p[0])) for p in poly], np.int32)
            cv2.polylines(out, [arr], False, 1, thickness=3)
            if polylines is not None:
                polylines.append([int(v) for v in arr.reshape(-1)])
    return out


def _split_mixed_colour(sub, core, cc, n, st):
    """Split the dominant blob of a label box by hue when it is several drawn things of different colour.

    Lettering that touches a line of another colour - blue "piedmont" sitting on a red fault line, with
    a grey patch edge through it - comes out of the difference-from-background test as ONE component.
    Its extent across the text line is then that of the line, it is rejected as "too big to be a
    character", and the glyph height is estimated from the specks that are left (6 px for 45 px text):
    the label is neither wiped nor measured. Colour tells the three apart where geometry cannot.

    Only the one dominant component is touched, and only when its pixels fall into hue groups that are
    genuinely different colours (a/b distance, lightness ignored). A single-colour word whose letters
    merged has one hue and is left exactly as it was.
    """
    if n <= 1:
        return cc, n, st
    big = 1 + int(np.argmax(st[1:, 4]))
    m = cc == big
    if st[big][4] < 0.4 * float(core.sum()) or st[big][4] < 200:
        return cc, n, st
    ab = cv2.cvtColor(sub, cv2.COLOR_BGR2LAB)[m][:, 1:].astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _c, lbl, ctr = cv2.kmeans(ab, 3, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    lbl = lbl.ravel()
    share = np.bincount(lbl, minlength=3) / float(lbl.size)
    sig = [k for k in range(3) if share[k] > 0.1]
    far = max((float(np.hypot(*(ctr[a] - ctr[b]))) for a in sig for b in sig if a < b), default=0.0)
    if len(sig) < 2 or far < 25.0:
        return cc, n, st
    parts = np.zeros(core.shape, np.int32)
    parts[m] = lbl + 1
    out = np.where(m, 0, cc).astype(np.int32)
    nxt = n
    for k in range(3):
        # Neutral parts (black frames, grey patch edges) are not this mask's business: dark neutral
        # lettering is removed by erase_text in the ink layer, and a neutral line kept here would be
        # wiped as if it were a letter (the frame around "nappe" was). They stay out, exactly as they
        # did while they were buried in the rejected blob.
        if float(np.hypot(ctr[k][0] - 128.0, ctr[k][1] - 128.0)) < 12.0:
            continue
        nk, ck = cv2.connectedComponents((parts == k + 1).astype(np.uint8), connectivity=8)
        if nk > 1:
            out[ck > 0] = ck[ck > 0] + nxt - 1
            nxt += nk - 1
    stats = np.zeros((nxt, 5), np.int32)
    for i in range(1, nxt):
        ys, xs = np.nonzero(out == i)
        if xs.size:
            stats[i] = [xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1, xs.size]
    return out, nxt, stats


def _own_hue(bgr, pen, rgb, S):
    """Pixels under the pen printed in the label's own (chromatic) colour, plus their anti-aliased rim."""
    ys, xs = np.nonzero(pen)
    out = np.zeros(pen.shape, bool)
    if rgb is None or max(rgb) - min(rgb) < 60 or xs.size == 0:
        return out
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    lab = _lab(bgr[y0:y1, x0:x1]).astype(np.float32)
    ref = _lab(np.array([[rgb[::-1]]], np.uint8)).astype(np.float32)[0, 0]
    own = ((np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0) > 30)
           & (np.hypot(lab[..., 1] - ref[1], lab[..., 2] - ref[2]) < 25)).astype(np.uint8)
    out[y0:y1, x0:x1] = cv2.dilate(own, np.ones((2 * S + 1, 2 * S + 1), np.uint8)) > 0
    return out


def _foreign_hue(bgr, pen, rgb):
    """Pixels under a label's wipe pen that are clearly drawn in ANOTHER colour than the label.

    A fault line running along a label passes the glyph-height test together with the letters it
    touches - across the text line it is exactly as tall as they are - so the whole run was wiped and
    the red fault through blue "fault-fold" came back as fragments. The label's own colour is known by
    now; strongly coloured pixels of a different hue are linework and stay. Only saturated pixels
    count, so the pale patch under the text and the anti-aliased glyph edges are wiped as before.
    """
    ys, xs = np.nonzero(pen)
    out = np.zeros(pen.shape, bool)
    if rgb is None or xs.size == 0:
        return out
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    lab = _lab(bgr[y0:y1, x0:x1]).astype(np.float32)
    ref = _lab(np.array([[rgb[::-1]]], np.uint8)).astype(np.float32)[0, 0]
    chroma = np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0)
    dab = np.hypot(lab[..., 1] - ref[1], lab[..., 2] - ref[2])
    out[y0:y1, x0:x1] = (chroma > 40) & (dab > 45)
    return out & pen


def glyph_mask_colour(bgr, labels, S, close_px=21, diff_thr=35, measure_only=False, split_for=(),
                      foreign_out=None, own_hue_only=False, on_line_only=False):
    """Pixels inside a label box that look like text, whatever colour that text is.

    The wipe runs on the ink mask, so a coloured label sitting on a coloured patch is never wiped at
    all - the palette calls it a "colour", not "ink" - and it comes back as ghost lettering traced
    into the patch underneath. This finds those pixels by estimating the text-free background with a
    closing wider than a stroke and taking the difference, then locking anything whose HEIGHT rules it
    out as a character: a stroke is flat, a patch is tall, text matches the glyph height. Judging by
    width instead would lock whole rotated words (one English word is many times wider than a
    character), which is the mistake that left fragments of every long label behind.

    Two jobs share this one pass over the glyph ink, hence `measure_only`:
      * always - fill in each label's 'measured' run (length, glyph height, centre), which the font
        sizing needs for labels an axis-aligned box cannot describe;
      * unless measure_only - also return the mask, so the caller can widen the wipe to those glyphs.
    """
    h2, w2 = bgr.shape[:2]
    out = np.zeros((h2, w2), np.uint8)
    kbg = np.ones((close_px, close_px), np.uint8)
    # Tried and rejected: dropping whole components that reach far outside the label box, to stop this
    # mask (which feeds an inpaint, with no _heal_crossings behind it) from eating fault lines. It does
    # remove 29% of the linework damage at what looked like no cost - but a label sitting ON a line
    # shares one component with it, so Longmenshan / fault-fold / nappe lost their wipe and came back
    # as coloured ghost lettering. Whole-component vetoes cannot work here: glyph and line are the same
    # object by then. Anything better has to separate them before they are joined, not after.
    for li, lab in enumerate(labels):
        box = lab.get('box')
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, w2 - 1), min(y1, h2 - 1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        # Bound everything to the detector's polygon when there is one: the axis-aligned box of a slanted
        # label covers 1.67x the text area, and the surplus is linework that must not be wiped.
        qm = None
        if QUAD_WIPE and lab.get('quad'):
            _f = np.zeros((h2, w2), np.uint8)
            cv2.fillPoly(_f, [np.array([[int(round(p[0])), int(round(p[1]))] for p in lab['quad']],
                                       np.int32)], 1)
            qm = cv2.dilate(_f, np.ones((2 * S + 1, 2 * S + 1), np.uint8)) > 0
        sub = bgr[y0:y1 + 1, x0:x1 + 1]
        bg = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, kbg)
        core = (np.abs(sub.astype(np.int16) - bg.astype(np.int16)).max(axis=2) > diff_thr).astype(np.uint8)
        core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        if qm is not None:
            core = core & qm[y0:y1 + 1, x0:x1 + 1].astype(np.uint8)
        if not core.any():
            continue
        h, w = core.shape
        n, cc, st, _ = cv2.connectedComponentsWithStats(core, 8)
        if li in split_for:
            cc, n, st = _split_mixed_colour(sub, core, cc, n, st)
        # Extent ACROSS the text line, not along the image axes.
        # A slanted label is the case this has to survive: its glyph blobs are axis-aligned objects in no
        # frame, so a run of slanted letters spans the box diagonally and the "2.4x glyph height" test
        # rejects every one of them. That emptied the mask - and an empty mask is what left slanted
        # labels BOTH unsized (so they were left blank) and unwiped (so their pixels survived as coloured
        # residue traced into the patch). Measuring each blob across its own text line keeps the test
        # meaningful at any angle. For near-horizontal labels this is exactly the old behaviour.
        _ar = math.radians(float(lab.get('angle') or 0.0))
        _across = None if abs(_ar) < 0.07 else (-math.sin(_ar), math.cos(_ar))

        def _ext(comp, cx_, cy_):
            """Blob extent across the text line, or None to keep using the axis-aligned height."""
            if _across is None:
                return None
            ys_i, xs_i = np.nonzero(comp)
            if xs_i.size == 0:
                return 0.0
            pv = (xs_i - cx_) * _across[0] + (ys_i - cy_) * _across[1]
            return float(pv.max() - pv.min())

        _eff = {}
        for i in range(1, n):
            if st[i][4] < 6:
                continue
            _e = _ext(cc == i, st[i][0] + st[i][2] / 2.0, st[i][1] + st[i][3] / 2.0)
            _eff[i] = float(st[i][3]) if _e is None else _e
        # Estimate the glyph height from the LARGEST components by area, not from all of them.
        # Slivers along a patch edge or a rule outnumber the glyphs in a busy box, and a median over
        # everything then lands on the 6 px floor - after which the 0.45..2.4x window below rejects the
        # real glyphs. Measured coverage of each label's own ink was 0.33 on average that way (Chengdu
        # 0.00, piedmont 0.05, fault-fold 0.02), which is exactly the coloured lettering that survives
        # as ghost text inside the patches. Taking the median of the top third BY AREA - the glyphs are
        # what dominates the area - lifts coverage to 0.83.
        _pairs = [(_eff[i], float(st[i][4])) for i in _eff]
        if _pairs:
            _pairs.sort(key=lambda t: -t[1])
            _top = _pairs[:max(3, len(_pairs) // 3)]
            glyph_h = float(np.median([hh for hh, _a in _top]))
        else:
            glyph_h = h * 0.3
        glyph_h = float(min(max(glyph_h, 6.0), 0.9 * h))
        # the median blob size across the text line: unlike the span-based g it is not moved by a line
        # running along the lettering, so it is what the font sizing uses to cap tracked-out labels
        lab['run_gh'] = round(glyph_h, 2)
        keep = np.zeros_like(core, bool)
        for i in range(1, n):
            _x, _y, ww, hh, _a = st[i]
            hh_e = _eff.get(i, float(hh))
            if hh_e < 0.45 * glyph_h or hh_e > 2.4 * glyph_h:
                continue                            # a flat stroke, or far too big to be a character
            if ww > 6.0 * glyph_h and hh_e < 0.7 * h:
                continue                            # spans the box and stays flat: a rule line
            keep |= cc == i
        if not keep.any():
            continue
        if on_line_only and _across is not None:
            # Pieces of a fault line in the corners of a slanted box pass the height test as well (their
            # extent across the line is a glyph's), and the glyph-height span then took them in:
            # Zhongjiang measured g = 99 for 25 px letters and was left blank. A slanted box is mostly
            # off the text line, so a piece is kept only if it lies ON the line the bulk of the ink
            # defines - the same rule pass 2 applies to what it adds.
            ids = [i for i in range(1, n) if keep[cc == i].any()]
            if len(ids) >= 3:
                off = np.array([(st[i][0] + st[i][2] / 2.0) * _across[0] + (st[i][1] + st[i][3] / 2.0)
                                * _across[1] for i in ids])
                wts = np.array([float(st[i][4]) for i in ids])
                o = np.argsort(off)
                med = off[o][np.searchsorted(np.cumsum(wts[o]), 0.5 * wts.sum())]
                for i, of_ in zip(ids, off):
                    if abs(of_ - med) > 0.9 * glyph_h:
                        keep[cc == i] = False
            if not keep.any():
                continue
        # --- pass 2: follow the text line past the box edges.
        # An OCR box is routinely shorter than the word it covers, and letters outside it sit inside no
        # pen at all - which is why every long label came back as fragments of its own tail. Track the
        # line the box already found and pick up glyph-shaped ink that continues it.
        ang = math.radians(float(lab.get('angle') or 0.0))
        ux, uy = math.cos(ang), math.sin(ang)          # along the line, image coordinates
        ys1, xs1 = np.nonzero(keep)
        cxm, cym = float(xs1.mean()) + x0, float(ys1.mean()) + y0
        pj = (xs1 - xs1.mean()) * ux + (ys1 - ys1.mean()) * uy
        half = float(max(abs(float(pj.min())), abs(float(pj.max())))) if pj.size else 0.0
        reach = 1.5 * glyph_h
        m = int(2 * glyph_h)
        # The search area must ALWAYS contain the box itself: slicing with a box that reaches outside it
        # raises, and a crashed layers step used to be masked by the stale file from the previous run.
        EX0 = max(min(x0, int(cxm - half - reach - m)), 0)
        EY0 = max(min(y0, int(cym - half - reach - m)), 0)
        EX1 = min(max(x1 + 1, int(cxm + half + reach + m)), w2)
        EY1 = min(max(y1 + 1, int(cym + half + reach + m)), h2)
        pen = np.zeros((h2, w2), bool)
        glyph_all = np.zeros((h2, w2), bool)
        glyph_all[y0:y1 + 1, x0:x1 + 1] = keep
        pen[y0:y1 + 1, x0:x1 + 1] = cv2.dilate(keep.astype(np.uint8), np.ones((3, 3), np.uint8),
                                               iterations=2) > 0
        if EX1 - EX0 > 8 and EY1 - EY0 > 8:
            sub2 = bgr[EY0:EY1, EX0:EX1]
            bg2 = cv2.morphologyEx(sub2, cv2.MORPH_CLOSE, kbg)
            core2 = (np.abs(sub2.astype(np.int16) - bg2.astype(np.int16)).max(axis=2) > diff_thr).astype(np.uint8)
            core2 = cv2.morphologyEx(core2, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
            n2, cc2, st2, _ = cv2.connectedComponentsWithStats(core2, 8)
            grow = np.zeros_like(core2, bool)
            for i in range(1, n2):
                cxx, cyy, ww, hh, _a = st2[i]
                # Same across-the-line rule as pass 1: pass 2 works in image coordinates too, so a
                # slanted label's blobs must not be judged by their axis-aligned height here either.
                if _across is None:
                    hh_e = float(hh)
                else:
                    _c = cc2 == i
                    _ys, _xs = np.nonzero(_c)
                    if _xs.size == 0:
                        continue
                    _pv = ((_xs - (cxx + ww / 2.0)) * _across[0]
                           + (_ys - (cyy + hh / 2.0)) * _across[1])
                    hh_e = float(_pv.max() - _pv.min())
                if hh_e < 0.45 * glyph_h or hh_e > 2.4 * glyph_h:
                    continue                          # a stroke, or far too big to be a character
                px, py = EX0 + cxx + ww / 2.0, EY0 + cyy + hh / 2.0
                dx, dy = px - cxm, py - cym
                if abs(dx * uy - dy * ux) > 0.9 * glyph_h:
                    continue                          # off the text line
                if abs(dx * ux + dy * uy) > half + reach:
                    continue                          # too far along it: a neighbouring label
                grow[cc2 == i] = True
            if grow.any():
                glyph_all[EY0:EY1, EX0:EX1] |= grow
                pen[EY0:EY1, EX0:EX1] |= cv2.dilate(grow.astype(np.uint8), np.ones((3, 3), np.uint8),
                                                    iterations=2) > 0
        # --- measure the run itself, so the font size stops depending on how big the box is.
        # fit_rotated_text() recovers the text length from the axis-aligned box, but that inversion is
        # singular at 45 degrees (det = cos 2theta) and falls back to max(W,H) - the size then follows
        # the box rather than the text, which is what turned every rotated label into a giant. Rotating
        # to the label's own angle and projecting the glyph ink takes the box out of the equation.
        if EY1 - EY0 > 8 and EX1 - EX0 > 8:
            gloc = np.zeros((EY1 - EY0, EX1 - EX0), bool)
            gloc[(y0 - EY0):(y1 + 1 - EY0), (x0 - EX0):(x1 + 1 - EX0)] = glyph_all[y0:y1 + 1, x0:x1 + 1]
            ys_m, xs_m = np.nonzero(gloc)
            # Height comes from the box interior only: the grown part runs along the text line and may
            # have picked up a rule or a neighbouring label, which inflated the glyph height badly.
            ys_k, xs_k = np.nonzero(keep)
            if xs_m.size >= 24 and xs_k.size >= 16:
                t = math.radians(float(lab.get('angle') or 0.0))
                mux, muy = math.cos(t), math.sin(t)
                mvx, mvy = -math.sin(t), math.cos(t)
                pu = xs_m * mux + ys_m * muy
                dxk = xs_k - float(xs_k.mean())
                dyk = ys_k - float(ys_k.mean())
                pvk = dxk * mvx + dyk * mvy

                def _span(v, frac=0.8):
                    """Width of the narrowest interval that holds `frac` of the samples.

                    max-min is what a few stray pixels ruin. Fragments of a line or a patch edge inside
                    the box stretched the measured glyph height to 115-230 px on labels whose text is
                    ~30 px tall, and the sanity test below then discarded the measurement - which is why
                    those labels came out blank on the sheet AND unwiped in the graphics. A trimmed span
                    ignores the strays while still covering the real run.
                    """
                    if v.size == 0:
                        return 0.0
                    s = np.sort(v)
                    k = max(1, int(frac * s.size))
                    if s.size <= k:
                        return float(s[-1] - s[0])
                    return float((s[k:] - s[:s.size - k]).min())

                meas_l = _span(pu)
                meas_g = _span(pvk)
                # A slanted box is mostly empty space, so a line crossing it inflates g without bound.
                # A real glyph height stays well under half the box's short side, and a word is by
                # definition longer than it is tall. A measurement failing either test is contaminated
                # by linework and is dropped, which sends that label to the blank-for-a-human path.
                short = min(x1 - x0, y1 - y0)
                if (meas_l >= 4.0 and meas_g >= 3.0 and meas_g <= 0.55 * short
                        and 1.4 <= meas_l / meas_g <= 15.0):
                    lab['measured'] = {
                        'L': round(meas_l, 2), 'g': round(meas_g, 2),
                        'cx': round(float(xs_m.mean()) + EX0, 1),
                        'cy': round(float(ys_m.mean()) + EY0, 1),
                    }
        mz = lab.get('measured')
        if mz and not own_hue_only:
            # Letters fused with a line of another colour are rejected whole by the shape test above
            # and survived as blobs (the "me" of red Longmenshan on the black frame line). The run is
            # measured by now, so every pixel of the label's own colour inside its text strip is a
            # glyph: take those too.
            strip = np.zeros((h2, w2), np.uint8)
            rect = ((mz['cx'], mz['cy']), (mz['L'] + mz['g'], 2.2 * mz['g']),
                    float(lab.get('angle') or 0.0))
            cv2.fillPoly(strip, [cv2.boxPoints(rect).astype(np.int32)], 1)
            pen |= _own_hue(bgr, strip > 0, lab.get('color'), S) & (strip > 0)
        if qm is not None:
            pen &= qm
        if foreign_out is not None:
            foreign_out |= _foreign_hue(bgr, pen, lab.get('color'))
        if own_hue_only:
            pen &= _own_hue(bgr, pen, lab.get('color'), S)
        lab['wipe_colour_px'] = int(pen.sum())
        if not measure_only:
            out[pen] = 1
    return out


def _chain_polylines(polylines, tol):
    """Join skeleton paths that meet end to end into longer polylines.

    The skeleton walk splits a line at every junction, so the outline around a few strata bands came out
    as 753 separate polylines, 462 of them under 10 px - each an object in the CDR. At a junction the two
    paths that continue each other most straightly are joined (a T keeps its stem separate); nothing is
    removed, only the number of pieces drops."""
    pls = [list(p) for p in polylines if len(p) >= 4]
    if len(pls) < 2:
        return pls

    def ends(p):
        return (p[0], p[1]), (p[-2], p[-1])

    def outdir(p, at_start):
        # direction pointing OUT of the polyline at that end, over its first ~3 vertices
        if at_start:
            a, b = (p[0], p[1]), (p[min(4, len(p) - 2)], p[min(5, len(p) - 1)])
        else:
            a, b = (p[-2], p[-1]), (p[max(len(p) - 6, 0)], p[max(len(p) - 5, 1)])
        dx, dy = a[0] - b[0], a[1] - b[1]
        n = math.hypot(dx, dy) or 1.0
        return dx / n, dy / n

    # candidate joints: endpoint pairs within tol (grid-bucketed), best continuation first, one join per end
    E = []                                        # (x, y, poly index, at_start)
    for i, p in enumerate(pls):
        E.append((p[0], p[1], i, True))
        E.append((p[-2], p[-1], i, False))
    grid = {}
    for k, (x, y, _i, _s) in enumerate(E):
        grid.setdefault((int(x // tol), int(y // tol)), []).append(k)
    pairs = []
    for a, (xa, ya, ia, sa) in enumerate(E):
        gx, gy = int(xa // tol), int(ya // tol)
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for b in grid.get((gx + ox, gy + oy), ()):
                    if b <= a:
                        continue
                    xb, yb, ib, sb = E[b]
                    if ia == ib or math.hypot(xa - xb, ya - yb) > tol:
                        continue
                    da, db = outdir(pls[ia], sa), outdir(pls[ib], sb)
                    straight = -(da[0] * db[0] + da[1] * db[1])      # 1 = the two continue each other
                    if straight > 0.5:
                        pairs.append((straight, a, b))
    pairs.sort(reverse=True)
    link = {}                                     # endpoint -> endpoint it is joined to
    parent = list(range(len(pls)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for _st, a, b in pairs:
        if a in link or b in link:
            continue
        ra, rb = root(E[a][2]), root(E[b][2])
        if ra == rb:
            continue                              # would close a loop of pieces
        link[a], link[b] = b, a
        parent[ra] = rb
    # walk each chain from a free end
    out, seen = [], set()

    def endpoint(i, at_start):
        return 2 * i + (0 if at_start else 1)
    for i in range(len(pls)):
        if i in seen:
            continue
        # find a free end of this chain
        cur, start_end = i, endpoint(i, True)
        visited = {i}
        while start_end in link:
            nxt = link[start_end]
            j, js = E[nxt][2], E[nxt][3]
            if j in visited:
                break
            visited.add(j)
            cur, start_end = j, endpoint(j, not js)
        # walk forward from start_end
        chain, e = [], start_end
        while True:
            j, js = E[e][2], E[e][3]
            if j in seen:
                break
            seen.add(j)
            p = pls[j] if js else [v for k in range(len(pls[j]) - 2, -1, -2) for v in pls[j][k:k + 2]]
            chain.extend(p if not chain else p[2:])
            far = endpoint(j, not js)
            if far not in link:
                break
            e = link[far]
        if chain:
            out.append(chain)
    return out


def line_strokes(mask, rgb):
    """Centerline strokes for a colour layer that is nothing but thin lines (fault lines), or None.

    Outline tracing (PowerTRACE's default) turns a 2 px red fault into a filled closed shape; with the
    patch rim added it rendered as a brown-edged sausage. CorelDRAW's own guidance is that maps and line
    drawings want CENTERLINE tracing - unfilled strokes - and the grey rules here are already redrawn
    that way. A layer qualifies when opening it with a disk wider than a line leaves almost nothing
    (it has no solid areas) and its area per centreline pixel - the mean line width - stays small.
    The skeleton walk is the one in cdr_trace_skeleton (Zhang-Suen thinning, node-split paths).
    """
    m = (mask > 0).astype(np.uint8)
    area = int(m.sum())
    if area < 40 * S:
        return None
    solid = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * S + 5, 2 * S + 5)))
    if solid.sum() > 0.1 * area:
        return None
    # Lines are LONG. A layer of specks is thin too - on the JPEG Tibet map the dark-olive rims and dots
    # of the small green patches passed the tests above, and as a "line" layer it was grown into the
    # paper and repainted near-white, wiping ten patches. The fault layer of the geology map has all of
    # its area in components over 15 px; that olive layer 17%.
    n_, _, st_, _ = cv2.connectedComponentsWithStats(m, 8)
    if n_ > 1:
        long_ = np.maximum(st_[1:, 2], st_[1:, 3]) >= 15 * S
        if st_[1:, 4][long_].sum() < 0.7 * area:
            return None
    import cdr_trace_skeleton as sk
    skel = sk.skeletonize(m * 255)
    length = int(skel.sum())
    if length == 0:
        return None
    width = area / float(length)
    if width > 3.5 * S:
        return None
    polylines = []
    for path in sk.trace_paths(skel):
        if len(path) < 5 * S:             # junction spurs; 5% of the rim length on the geology section
            continue
        ap = sk._simplify(path, 0.7 * S).reshape(-1, 2)
        polylines.append([float(v) for xy in ap for v in xy])
    polylines = _chain_polylines(polylines, 2.0 * S)
    if not polylines:
        return None
    return {'polylines': polylines, 'width_px': round(max(width, 1.0), 2), 'color': list(rgb)}


def dashed_strokes(mask, rgb):
    """Straight dashed lines of a colour layer as vector strokes: (strokes, rest_mask) or None.

    line_strokes() only takes LONG lines, so a layer of dashed faults (each dash ~15 px) fell through to
    outline tracing: one filled shape per dash, 243 of them on a simple geological section, where the
    drawing has fifteen lines. Here collinear dashes are chained into one straight line each, drawn later
    as a single stroke with a dash pattern. Pieces no chain explains (a stray dot, a curved rim) come back
    in rest_mask and are traced as before, so nothing is dropped. The layer qualifies only if the chains
    account for most of its pixels - a scatter of short strokes is not a set of dashed lines.
    """
    m = (mask > 0).astype(np.uint8)
    area = int(m.sum())
    if area < 60 * S * S:
        return None
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 7:
        return None
    dashes = []                                   # (comp, cx, cy, ux, uy, length, width, area)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 4 * S * S or max(w, h) > 60 * S:
            continue
        ys, xs = np.nonzero(lab[y:y + h, x:x + w] == i)
        xs = xs + x
        ys = ys + y
        cx, cy = float(xs.mean()), float(ys.mean())
        ev, evec = np.linalg.eigh(np.cov(np.vstack([xs - cx, ys - cy])))
        ln, wd = 4 * math.sqrt(max(ev[1], 1e-6)), 4 * math.sqrt(max(ev[0], 1e-6))
        dashes.append((i, cx, cy, float(evec[0, 1]), float(evec[1, 1]), ln, wd, int(a)))
    if len(dashes) < 6:
        return None
    P = np.array([[d[1], d[2]] for d in dashes])
    U = np.array([[d[3], d[4]] for d in dashes])
    used, lines = set(), []
    for j in sorted(range(len(dashes)), key=lambda q: -dashes[q][5]):
        if j in used or dashes[j][5] < 2.0 * dashes[j][6]:
            continue                              # a dot gives no direction to seed a line with
        ux, uy = dashes[j][3], dashes[j][4]
        ox, oy = dashes[j][1], dashes[j][2]
        members = [j]
        for _ in range(3):                        # grow, refit the axis on the members, grow again
            rel = P - [ox, oy]
            t = rel @ [ux, uy]
            off = np.abs(rel @ [-uy, ux])
            para = np.abs(U @ [ux, uy])
            cand = [q for q in range(len(dashes)) if q == j or (
                q not in used and off[q] <= 3.0 * S and para[q] > 0.9)]
            order = sorted(cand, key=lambda q: t[q])
            si = order.index(j)
            maxgap = 3.0 * max(float(np.median([dashes[q][5] for q in order])), 4.0 * S)
            lo = hi = si
            while lo > 0 and t[order[lo]] - t[order[lo - 1]] <= maxgap:
                lo -= 1
            while hi < len(order) - 1 and t[order[hi + 1]] - t[order[hi]] <= maxgap:
                hi += 1
            members = order[lo:hi + 1]
            if len(members) < 2:
                break
            c0 = P[members].mean(axis=0)
            _, _, vt = np.linalg.svd(P[members] - c0)
            ux, uy = float(vt[0][0]), float(vt[0][1])
            ox, oy = float(c0[0]), float(c0[1])
        if len(members) < 3:
            continue
        spans, lens = [], []
        for q in members:
            i = dashes[q][0]
            x, y, w, h, a = st[i]
            ys, xs = np.nonzero(lab[y:y + h, x:x + w] == i)
            tq = (xs + x - ox) * ux + (ys + y - oy) * uy
            spans.append((float(tq.min()), float(tq.max())))
            lens.append(float(tq.max() - tq.min() + 1))
        spans.sort()
        gaps = [spans[q + 1][0] - spans[q][1] - 1 for q in range(len(spans) - 1)]
        gaps = [g for g in gaps if g > 0]
        t0, t1 = spans[0][0], spans[-1][1]
        used.update(members)
        lines.append({'pl': [ox + ux * t0, oy + uy * t0, ox + ux * t1, oy + uy * t1],
                      'dash': float(np.median(lens)), 'gap': float(np.median(gaps)) if gaps else 0.0,
                      'width': float(np.median([dashes[q][7] / max(lens[k], 1.0) for k, q in enumerate(members)])),
                      'comps': [dashes[q][0] for q in members]})
    if len(lines) < 3:
        return None
    # one drawing, one fault style: chains whose width or rhythm disagrees with the majority are a
    # rim or a frame edge that happened to line up (sea-floor outline: width 1.0, gap 52 vs 2.6 / 3)
    mw = float(np.median([l['width'] for l in lines]))
    md = float(np.median([l['dash'] for l in lines]))
    mg = float(np.median([l['gap'] for l in lines]))
    lines = [l for l in lines if 0.6 * mw <= l['width'] <= 1.6 * mw and l['gap'] <= max(2.0 * mg, 0.6 * md)]
    if len(lines) < 3:
        return None
    covered = np.isin(lab, [i for l in lines for i in l['comps']])
    if float(covered.sum()) < 0.6 * area:
        return None
    rest = (m.astype(bool) & ~covered).astype(np.uint8)
    strokes = {'polylines': [[round(v, 1) for v in l['pl']] for l in lines], 'width_px': round(mw, 2),
               'color': list(rgb), 'dash': [round(md, 1), round(max(mg, 1.0), 1)]}
    return strokes, rest


def drop_boundary_slivers(masks, others, bgr):
    """Remove the thin crumbs a flat-colour drawing leaves in its colour layers, in place.

    Where two flat regions meet, the anti-aliased pixels have an in-between colour, and they land in
    whatever cluster is nearest: between red and yellow that is orange, so the orange shale layer of a
    geological section came out as 2 real bands plus 500 slivers, each traced as its own curve (2274
    curves for a drawing of ~200 objects). A sliver is thin (nothing survives a 3x3 erosion), small, and
    sits ON A BOUNDARY: its ring touches at least two other regions. A real small patch - a JPEG-faded
    map patch, a symbol - is surrounded by one colour and stays. Only layers made of solid areas are
    cleaned; in a line layer the thin pieces are the content. Freed pixels go to the neighbouring region
    that took over, so no white seam opens where a sliver was. Returns the number of slivers removed.

    Thin is not enough: the red wavy arrows of the same section are thin, lie on the tan/rim boundary and
    were repainted yellow. A sliver's colour is a BLEND of the two regions it separates (orange between
    red and yellow); a real thin element has its own colour, far from every such blend.
    """
    keys = list(masks)
    if not keys:
        return 0
    h, w = next(iter(masks.values())).shape
    owner = np.zeros((h, w), np.int32)            # 0 = paper, 1.. = colour layers, -1.. = others
    for j, k in enumerate(keys):
        owner[masks[k] > 0] = j + 1
    for j, o in enumerate(others):
        owner[(o > 0) & (owner == 0)] = -(j + 1)
    col = {0: np.array([255.0, 255.0, 255.0])}
    for j, k in enumerate(keys):
        px = bgr[masks[k] > 0]
        col[j + 1] = np.median(px, axis=0) if len(px) else col[0]
    for j, o in enumerate(others):
        px = bgr[(o > 0) & (owner == -(j + 1))]
        col[-(j + 1)] = np.median(px, axis=0) if len(px) else col[0]

    def is_blend(c, ids_):
        for p_ in range(len(ids_)):
            for q_ in range(p_ + 1, len(ids_)):
                a_, b_ = col[int(ids_[p_])], col[int(ids_[q_])]
                ab = b_ - a_
                t_ = float(np.dot(c - a_, ab) / max(float(np.dot(ab, ab)), 1.0))
                if 0.05 <= t_ <= 0.95 and float(np.linalg.norm(a_ + t_ * ab - c)) < 30.0:
                    return True
        return False
    k3 = np.ones((3, 3), np.uint8)
    ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * S + 3, 4 * S + 3))
    solid = [k for k in keys if masks[k].any() and float(cv2.erode((masks[k] > 0).astype(np.uint8), ell).sum())
             >= 0.3 * float((masks[k] > 0).sum())]
    freed = np.zeros((h, w), bool)
    removed = 0
    for j, k in enumerate(keys):
        m = (masks[k] > 0).astype(np.uint8)
        a_all = int(m.sum())
        if a_all == 0 or float(cv2.erode(m, ell).sum()) < 0.3 * a_all:
            continue                              # not a layer of solid areas
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        core = cv2.erode(m, k3)
        for i in range(1, n):
            x, y, bw, bh, a = st[i]
            if a >= 150 * S * S:
                continue
            x0, y0, x1, y1 = max(x - 2, 0), max(y - 2, 0), min(x + bw + 2, w), min(y + bh + 2, h)
            comp = lab[y0:y1, x0:x1] == i
            ring = (cv2.dilate(comp.astype(np.uint8), k3) > 0) & ~comp
            ids, cnt = np.unique(owner[y0:y1, x0:x1][ring], return_counts=True)
            if int(core[y0:y1, x0:x1][comp].sum()) > max(2, 0.1 * a):
                continue                          # has a body: a real small element
            sel = (ids != j + 1) & (cnt >= max(1, 0.15 * cnt.sum()))
            # 1-3 px specks go even inside one colour (a faint dotted line in the sandstone came out as
            # orange dots); the smallest real patch kept elsewhere (clean_mask) is S*S with a body
            own = bgr[y0:y1, x0:x1][comp].astype(np.float64).mean(axis=0)   # the sliver's real colour
            if a >= 4 * S * S and (int(sel.sum()) < 2 or not is_blend(own, ids[sel])):
                continue                          # one colour around it, or a colour of its own: keep
            m[y0:y1, x0:x1][comp] = 0
            freed[y0:y1, x0:x1] |= comp
            removed += 1
        masks[k] = m
    # Seams: pixels no layer owns but that lie INSIDE the drawing (enclosed, small) showed as white
    # hairlines along every band and as a white hole where a patch had been. Paper proper - the margin,
    # a legend's background - is one large unowned area and stays. Freed sliver pixels join the seams.
    owned = owner != 0
    for k in keys:
        owned |= masks[k] > 0
    n_, lab_, st_, _ = cv2.connectedComponentsWithStats((~owned).astype(np.uint8), 4)
    seam = np.zeros((h, w), bool)
    for i in range(1, n_):
        x, y, bw, bh, a = st_[i]
        touches = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if not touches and a <= 400 * S * S:
            seam[y:y + bh, x:x + bw] |= lab_[y:y + bh, x:x + bw] == i
    # hand freed and seam pixels to the adjacent solid colour layer, a ring at a time, largest first
    left = freed | seam
    for _ in range(6 * S):
        if not left.any():
            break
        for k in sorted(solid, key=lambda k_: -int(masks[k_].sum())):
            grow = (cv2.dilate(masks[k], k3) > 0) & left
            if grow.any():
                masks[k] = (masks[k].astype(bool) | grow).astype(np.uint8)
                left &= ~grow
    return removed


def _left_square(src_gray, box, S, alone=False, gray_all=None):
    """Right edge of a hollow square symbol at the left end of a horizontal label box, or None.

    A square outline has ink in all four corners of its bounding box and a large rectangular hole;
    the round letters that also enclose a hole (O, Q, o, d) leave their bbox corners empty, which
    is what kept "Qinling" from losing its Q."""
    # judged on the SOURCE: in the SR ink mask the little squares come out filled (Luojiang, Hui)
    x0, y0, x1, y1 = (int(v // S) for v in box)
    bh = y1 - y0
    if bh < 6 or (not alone and x1 - x0 < 2 * bh):
        return None
    sub = (src_gray[y0:y1 + 1, x0:x1 + 1] < 140).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(sub, 8)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if (not alone and x > 0.3 * (x1 - x0)) or h < 0.4 * bh or h > 1.1 * bh or h < 6:
            continue
        if not 0.75 <= w / float(h) <= 1.33:
            continue
        comp = (cc[y:y + h, x:x + w] == i).astype(np.uint8)
        filled = np.zeros_like(comp)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(filled, cnts, -1, 1, -1)
        if (filled.sum() - comp.sum()) < 0.35 * filled.sum():
            continue                                    # not hollow
        c = max(1, min(w, h) // 5)
        corners = [comp[:c, :c], comp[:c, -c:], comp[-c:, :c], comp[-c:, -c:]]
        if all(k.mean() > 0.3 for k in corners):
            return int((x0 + x + w) * S)
    # The outline test above needs the square as a component of its own. A fault line cutting it
    # (Xindu) or a frame line touching it (Gaomiao) breaks that, but the square's HOLE survives: a
    # small closing mends a 1-2 px cut, and a rectangular hole (a letter's hole is round: o, a, D fill
    # 0.78-0.85 of their box) of about glyph size at the left end is the symbol.
    # (on the plain grey here: a coloured line over the square's edge completes its ring)
    if gray_all is not None:
        sub = (gray_all[y0:y1 + 1, x0:x1 + 1] < 140).astype(np.uint8)
    closed = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, cc, st, _ = cv2.connectedComponentsWithStats(1 - closed, 4)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if x == 0 or y == 0 or x + w >= sub.shape[1] or y + h >= sub.shape[0]:
            continue
        if (not alone and x > 0.3 * (x1 - x0)) or h < 0.3 * bh or h > bh or h < 4:
            continue
        if not 0.7 <= w / float(h) <= 1.4 or a < 0.88 * w * h:
            continue
        ring = closed[y + h // 2, x + w:min(x + w + max(3, h // 3), sub.shape[1])]
        t = int(np.argmin(ring)) if (ring == 0).any() else len(ring)
        return int((x0 + x + w + t) * S)
    return None


def text_colour(src_bgr, box):
    """The colour a label is printed in, sampled on the source from its separated foreground.

    The old sample looked only at INK components, so a coloured label - blue on a blue basin, red along
    a nappe - had no pixels to sample and fell back to black: almost every coloured label of the
    geology map came out black. Layered text-editing pipelines (detect -> segment foreground -> restore
    background -> redraw) take the colour from the segmented foreground instead, and so does this: the
    pixels that stand out from a closing-estimated background, minus pieces running out of the box
    (a line crossing it). Returns RGB, or None when the box holds too little foreground to judge;
    the caller keeps its own ink sample for neutral lettering (see there).
    """
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    H, W = src_bgr.shape[:2]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W - 1), min(y1, H - 1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    p = 12
    X0, Y0, X1, Y1 = max(x0 - p, 0), max(y0 - p, 0), min(x1 + p, W), min(y1 + p, H)
    win = src_bgr[Y0:Y1, X0:X1]
    sub = win.astype(np.int16)
    bg = cv2.morphologyEx(win, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8)).astype(np.int16)
    diff = np.abs(sub - bg).max(axis=2)
    core = (diff > 40).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(core, 8)
    keep = np.zeros(core.shape, bool)
    edge = np.zeros(core.shape, bool)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < 3:
            continue
        if x <= 0 or y <= 0 or x + w >= core.shape[1] or y + h >= core.shape[0]:
            edge |= cc == i
            continue
        keep |= cc == i
    # A line crossing the lettering fuses with it into one component that runs out of the window, and
    # dropping it threw the whole label away: blue "fault-fold" on a red fault kept only the black
    # dashes nearby and came out black. When the in-box pieces are the minor part, take the fused
    # component's pixels inside the box too; the colour clustering below separates line from glyphs.
    inner = np.zeros(core.shape, bool)
    inner[y0 - Y0:y1 - Y0 + 1, x0 - X0:x1 - X0 + 1] = True
    edge &= inner
    if int(edge.sum()) > 2 * int(keep.sum()):
        keep |= edge
    if int(keep.sum()) < 8:
        return None
    px = sub[keep].astype(np.float32)
    lab = cv2.cvtColor(win, cv2.COLOR_BGR2LAB)[keep].astype(np.float32)
    chroma = np.hypot(lab[:, 1] - 128.0, lab[:, 2] - 128.0)
    colourful = chroma > 25
    if colourful.mean() >= 0.3:
        # coloured lettering: the median of its coloured pixels, so a black frame or fault line
        # sharing the box cannot drag a blue label to black; a coloured line through a coloured label
        # (red fault through blue text) is split off by hue - the lettering is the larger cluster
        cp, ab = px[colourful], lab[colourful][:, 1:3]
        if len(cp) >= 20:
            _, lbl, _ = cv2.kmeans(ab.astype(np.float32), 2, None,
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 3,
                                   cv2.KMEANS_PP_CENTERS)
            lbl = lbl.ravel()
            big = int(np.bincount(lbl, minlength=2).argmax())
            ca, cb = ab[lbl == big].mean(0), ab[lbl != big].mean(0) if (lbl != big).any() else None
            if cb is not None and np.hypot(*(ca - cb)) > 25:
                cp = cp[lbl == big]
        c = np.median(cp, axis=0)
    else:
        # black or grey lettering: small strokes are 1-2 px and anti-aliased, so no pixel reaches the
        # true ink and a median comes out grey (西宁 122 for black print); the darkest tenth is the ink
        dark = px[np.argsort(px.sum(axis=1))[:max(1, len(px) // 10)]]
        c = dark.mean(axis=0)
    return [int(c[2]), int(c[1]), int(c[0])]


def colour_core(bgr, close_px=21, diff_thr=35):
    """Everything that is not flat background: linework, patch edges and lettering alike.

    Same test glyph_mask_colour uses per box, run on the whole image - a closing wider than a stroke
    estimates the background, and what stands out from it is drawn content."""
    bg = cv2.morphologyEx(bgr, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8))
    core = (np.abs(bgr.astype(np.int16) - bg.astype(np.int16)).max(axis=2) > diff_thr).astype(np.uint8)
    return cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0


def heal_colour_crossings(bgr_orig, bgr, wiped, labels, glyph_ink=None):
    """Put back the lines and patch edges the colour wipe cut through.

    erase_text answers "the wipe ate a line" with _heal_crossings: wipe conservatively, then reconnect
    what was cut. The colour wipe never had that second half - it hands its mask straight to an inpaint,
    so a fault line crossing a label box is gone for good. Every attempt to make the MASK smarter fails
    on the same wall: inside the box a glyph and the line it sits on are one connected object, and
    dropping that object un-wipes the label (measured: Longmenshan / fault-fold / nappe came back as
    coloured ghost lettering). Healing sidesteps it entirely - it never has to tell glyph from line,
    it only needs the line to still exist on BOTH sides of the gap, which a glyph never does.

    Feeds _heal_crossings binary stand-ins so the author's tested geometry (stub pairing, collinearity,
    "was this dark in the original") is reused rather than reinvented.
    """
    structure = colour_core(bgr_orig)
    img = np.where(structure, 0, 255).astype(np.uint8)                    # the drawing before the wipe
    # Only structure OUTSIDE the label boxes may act as a stub. _heal_crossings joins collinear stub
    # pairs, and the letters of a word are collinear by construction - the ink wipe never meets that
    # case because it clears a whole box at once and leaves no letter behind, but this mask is glyph
    # shaped and imperfect, so its leftovers seeded bridges and drew rows of letters back in (seen on
    # piedmont). A line crossing a box has ends on both sides of it; a leftover letter does not.
    inside = np.zeros(structure.shape, bool)
    for lab in labels:
        x0, y0, x1, y1 = (int(v) for v in lab['box'])
        inside[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] = True
    out = np.where(structure & (wiped == 0) & ~inside, 0, 255).astype(np.uint8)
    # _heal_crossings looks for stubs entering one window, so the window has to match the damage.
    # A label box is the right window for the ink wipe, which cuts one box-sized hole; the colour
    # wipe instead nibbles a line wherever a glyph crossed it, leaving many small gaps inside the
    # box that no box-sized window resolves (per-box healing alone recovered 4.8%, per-gap 12.1%).
    # So each gap is offered as its own window, then the boxes, then the gaps again - bridging one
    # gap creates the stub that lets the next one pair up. It converges after that.
    n, _, st, _ = cv2.connectedComponentsWithStats(wiped, 8)
    gaps = [{'box': [int(st[i][0]), int(st[i][1]), int(st[i][0] + st[i][2]), int(st[i][1] + st[i][3])]}
            for i in range(1, n)]
    healed = out.copy()
    for window in (gaps, labels, gaps):
        healed = _heal_crossings(img, healed, window)
    # only pixels the wipe actually took: everything else in bgr is already the original
    back = (healed < 200) & (out >= 200) & (wiped > 0)
    if glyph_ink is not None:
        # A bridge runs straight between two stubs, and a fault line often runs almost parallel to the
        # label it crosses - so the path grazes the lettering and restored it as a dotted trail along
        # the old baseline. erase_text already ruled those exact pixels to be this label's glyphs, so
        # they are never a line worth putting back, whatever the geometry says.
        back &= ~glyph_ink
    # The same trail in COLOURED lettering is invisible to that test - coloured glyphs are not in the
    # ink layer, which is the very reason the colour mask exists. But glyph_mask_colour already
    # measured where each label's lettering runs, so the band it occupies is known and nothing inside
    # it is ever restored. No glyph-versus-line decision is needed: the band is simply off limits.
    band = np.zeros(structure.shape, np.uint8)
    for lab in labels:
        m = lab.get('measured') or {}
        if m:
            t = math.radians(float(lab.get('angle') or 0.0))
            ux, uy = math.cos(t), math.sin(t)
            half_l, half_g = float(m['L']) / 2 + float(m['g']), float(m['g'])
            cx, cy = float(m['cx']), float(m['cy'])
            pts = [(cx + ux * a - (-uy) * b, cy + uy * a - ux * b)
                   for a, b in ((-half_l, -half_g), (half_l, -half_g), (half_l, half_g), (-half_l, half_g))]
            cv2.fillPoly(band, [np.array([[int(round(p)), int(round(q))] for p, q in pts], np.int32)], 1)
        else:
            x0, y0, x1, y1 = (int(v) for v in lab.get('tight', lab['box']))
            band[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] = 1
    back &= band == 0
    n = int(back.sum())
    if n:
        bgr = bgr.copy()
        bgr[back] = bgr_orig[back]
    return bgr, n


def _run_ratio(lab):
    """(L/g) per character of a label's measured run, or None - a font constant, not a size."""
    m = lab.get('measured') or {}
    text = lab.get('text') or ''
    n_cjk = sum(1 for ch in text if '⺀' <= ch <= '￯')
    n_eff = n_cjk + 0.5 * max(len(text) - n_cjk, 0)
    if not m or n_eff <= 0 or not m.get('g'):
        return None
    return float(m['L']) / float(m['g']) / n_eff


def glyph_mask_colour_best(bgr, labels, S, measure_only=False, foreign_out=None):
    """glyph_mask_colour, with the hue split applied only to the labels it demonstrably repairs.

    Splitting by hue rescues lettering fused to a line of another colour, but on a box holding many
    colours it can also hand the measurement a rim or a line fragment (Longmenshan's glyph height went
    from 32 to 115). The OCR text says how long a run should be for its height: over the 47 horizontal
    labels of the geology map (L/g)/characters spans 1.06-3.06. A label is split only when its plain
    measurement is outside that range (or missing) and the split one falls inside it; a label already
    measured sensibly is never touched. Returns (mask, number of labels split).
    """
    def ok(r):
        return r is not None and 1.0 <= r <= 3.1
    plain = [dict(l) for l in labels]
    glyph_mask_colour(bgr, plain, S, measure_only=True)
    trial = [dict(l) for l in labels]
    glyph_mask_colour(bgr, trial, S, measure_only=True, split_for=range(len(labels)))
    chosen = {i for i, (a, b) in enumerate(zip(plain, trial))
              if a.get('text') and not ok(_run_ratio(a)) and ok(_run_ratio(b))}
    mask = glyph_mask_colour(bgr, labels, S, measure_only=measure_only, split_for=chosen,
                             foreign_out=foreign_out)
    # Rescue only: a label still unmeasured is measured again keeping only the ink ON its text line.
    # Applied to every label this also moved good measurements (piedmont's fused glyph+fault run
    # dragged the line centre off the text), so it may only fill in what is missing, and only with a
    # run whose length per character is plausible.
    miss = [i for i, l in enumerate(labels) if l.get('text') and not l.get('measured')]
    if miss:
        again = [dict(labels[i]) for i in miss]
        glyph_mask_colour(bgr, again, S, measure_only=True, on_line_only=True)
        for i, l in zip(miss, again):
            if l.get('measured') and ok(_run_ratio(l)):
                labels[i]['measured'] = l['measured']
    if not measure_only:
        # The WIPE takes the split glyphs of every label, not only of those whose measurement it
        # repairs: a red "n" of Longmenshan fused with the black frame line, or the blue "c" of
        # Xinchang fused with a boundary, is rejected whole by the shape test and survived as a
        # blob. Only pixels in the label's own colour are added, so a rim or a line fragment that
        # the split mis-measures cannot widen the wipe.
        mask |= glyph_mask_colour(bgr, [dict(l) for l in labels], S, split_for=range(len(labels)),
                                  own_hue_only=True)
    return mask, len(chosen)


def color_masks(bgr, idx, centers, kinds, ink_dil, min_area):
    """Masks of the colour layers that survive (big enough, not an ink halo). Small patches in the
    SR image are mostly their darker outline, whose pixels fall into dark/brown clusters that are
    then dropped as too small, so the patch vanished. Pixels that are clearly chromatic but not in
    a kept colour are therefore re-assigned by hue (a/b, lightness weighted low) to the nearest
    kept colour, and each mask is closed so outline + fill become one solid patch."""
    h2, w2 = idx.shape
    kept = []
    for i, kind in enumerate(kinds):
        if kind != 'color':
            continue
        m = clean_mask(idx == i, min_area)
        if not m.any() or not big_enough(m, min_area, h2 * w2):
            continue
        if float((m & ink_dil).sum()) / float(m.sum()) > 0.6:
            continue            # anti-aliasing halo: a "colour" that only ever hugs the ink
        kept.append(i)
    if not kept:
        return {}
    lab = _lab(bgr).astype(np.float32)
    chroma = np.hypot(lab[..., 1] - 128, lab[..., 2] - 128)
    paper = bgr.min(axis=2) >= WHITE_MIN
    d = np.stack([np.hypot(lab[..., 1] - centers[i][1], lab[..., 2] - centers[i][2])
                  + 0.25 * np.abs(lab[..., 0] - centers[i][0]) for i in kept], axis=0)
    near, dist = d.argmin(axis=0), d.min(axis=0)
    # never pull in pixels of grey/pastel "grid" clusters: those are rules or flat fills handled on their own
    # (pale blue flowchart boxes were swallowed by the pale green layer, which then came out blue)
    # Only inside SOLID pale areas though: small JPEG-faded map patches also land in those clusters and
    # still need the hue rescue (excluding all of them lost 2-3 patches per faded map).
    grid_idx = [i for i, k in enumerate(kinds) if k == 'grid']
    pale = np.isin(idx, grid_idx).astype(np.uint8)
    pale_solid = cv2.morphologyEx(pale, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * S + 1, 4 * S + 1)))
    pale_solid = (cv2.dilate(pale_solid, np.ones((2 * S + 1, 2 * S + 1), np.uint8)) & pale) > 0
    loose = (~paper) & (chroma > 10) & ~np.isin(idx, kept) & ~pale_solid & (dist < 55)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out = {}
    for j, i in enumerate(kept):
        m = ((idx == i) | (loose & (near == j))).astype(np.uint8)
        m = clean_mask(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k3), min_area, chroma=chroma)
        if m.any():
            out[i] = m
    return out


def fill_text_holes(mask, label_zone, max_hole):
    """Close holes a colour region keeps where a label was wiped. Inpainting leaves glyph-shaped gaps; traced
    and outlined they rendered as a ghost of the old text ("质量合格??" and dots on a flowchart diamond).
    Only holes touching a label zone, or tiny ones anywhere, are filled - real holes (a white island inside
    a patch) are larger and away from text."""
    m = (mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask
    solid = np.zeros_like(m)
    cv2.drawContours(solid, cnts, -1, 1, -1)
    holes = solid & (1 - m)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(holes, 8)
    out = m.copy()
    for j in range(1, n):
        x, y, w, h, a = st[j]
        near_text = label_zone[y:y + h, x:x + w][lbl[y:y + h, x:x + w] == j].any()
        if a <= 4 * S * S or (near_text and a <= max_hole):
            out[y:y + h, x:x + w][lbl[y:y + h, x:x + w] == j] = 1
    return out


def has_solid_element(mask, min_core=6):
    """True if the mask contains at least one solid blob (survives erosion by S px, core >= min_core*S*S).
    A colour used once (a highlighted dot, one bar) is small in total area but solid; JPEG colour noise and
    anti-aliasing rims are thin or scattered and have no core."""
    core = cv2.erode((mask > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * S + 1, 2 * S + 1)))
    if not core.any():
        return False
    n, _, st, _ = cv2.connectedComponentsWithStats(core, 8)
    return bool(n > 1 and st[1:, 4].max() >= min_core * S * S)


def big_enough(mask, min_area, total_px):
    """Layer keep rule: large total area, or small but containing a solid element."""
    a = int((mask > 0).sum())
    if a < min_area:
        return False
    return a >= 0.0004 * total_px or has_solid_element(mask)


def clean_mask(mask, min_area, chroma=None):
    """Drop components below min_area. With `chroma`, a small piece that is clearly coloured is kept:
    in small/downscaled drawings real map patches are only 4-15 working px and were all dropped;
    JPEG colour noise is weakly chromatic and still goes."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros_like(mask, np.uint8)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        keep = a >= min_area
        if not keep and chroma is not None and a >= S * S:
            comp = lab[y:y + h, x:x + w] == i
            keep = float(chroma[y:y + h, x:x + w][comp].mean()) >= 25
        if keep:
            out[y:y + h, x:x + w][lab[y:y + h, x:x + w] == i] = 1
    return out


def build_layers(work_dir, confirmed, colors=6, min_region_px=None):
    color_png = os.path.join(work_dir, 'sr2_color.png')
    src_bgr = imread(os.path.join(work_dir, 'src.png'), cv2.IMREAD_COLOR)
    bgr = imread(color_png, cv2.IMREAD_COLOR)
    h2, w2 = bgr.shape[:2]
    w, h = w2 // S, h2 // S
    centers = extract_palette(bgr, k=colors)
    idx = assign_palette(bgr, centers)
    kinds = [classify(c) for c in centers]
    min_area = min_region_px if min_region_px is not None else 4 * S * S

    ink_mask = np.zeros((h2, w2), np.uint8)
    for i, kind in enumerate(kinds):
        if kind == 'ink':
            ink_mask |= (idx == i).astype(np.uint8)
    ink_mask = thicken_ink(bgr, ink_mask)
    ink_dil0 = cv2.dilate(ink_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * S + 3, 2 * S + 3)))
    color_px = np.zeros((h2, w2), bool)
    cmasks_pre = color_masks(bgr, idx, centers, kinds, ink_dil0, min_area)
    for m_ in cmasks_pre.values():
        color_px |= m_ > 0
    ink_mask = drop_region_rims(ink_mask, color_px)

    # labels live in the ink layer: erase them there, then re-create them as text objects
    if isinstance(confirmed, str):
        confirmed = json.load(open(confirmed, encoding='utf-8'))
    cand_path = os.path.join(work_dir, 'ocr_candidates.json')
    cands = json.load(open(cand_path, encoding='utf-8')).get('candidates', []) if os.path.exists(cand_path) else []

    def _match(box_src):
        """Angle and detector polygon for a confirmed label, from the nearest OCR candidate by IoU."""
        best_a, best_q, bi = 0.0, None, 0.0
        for cd in cands:
            b = cd['box']
            ix = max(0, min(box_src[2], b[2]) - max(box_src[0], b[0]))
            iy = max(0, min(box_src[3], b[3]) - max(box_src[1], b[1]))
            inter = ix * iy
            u = (box_src[2]-box_src[0])*(box_src[3]-box_src[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9
            if inter/u > bi:
                bi, best_a, best_q = inter/u, float(cd.get('angle', 0.0)), cd.get('quad')
        if bi <= 0.2:
            return 0.0, None
        return best_a, best_q

    ink_gray = np.where(ink_mask > 0, 0, 255).astype(np.uint8)
    ink_nh = ink_mask & (1 - _rule_mask(ink_mask))
    labels = []
    src_gray0 = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    src_gray_all = src_gray0.copy()
    # the place-name squares are black: a red fault running into one (Gaomiao) must not join it
    src_gray0[(src_bgr.max(axis=2).astype(np.int16) - src_bgr.min(axis=2)) > 60] = 255
    for c in confirmed or []:
        text = str(c['text']).strip()
        erase_only = bool(c.get('erase_only'))
        if not text and not erase_only:
            continue
        box = [int(v) * S for v in c['box']]
        box = [max(box[0], 0), max(box[1], 0), min(box[2], w2 - 1), min(box[3], h2 - 1)]
        auto_ang, quad_src = _match([int(v) for v in c['box']])
        ang = float(c['angle']) if 'angle' in c else auto_ang
        # A tall box read with angle 0 is rotated Latin text: the detector's polygon has a short flat
        # top edge, so its slant comes out 0, but RapidOCR turns any crop 1.5x taller than wide by 90
        # degrees counter-clockwise before reading it - a correct read therefore means the text runs
        # bottom-to-top, which is -90 here. Laid flat instead, "Longquanshan tectonic belt" became a
        # one-line sliver and a misread vertical coordinate a giant "0". (CJK is excluded: upright
        # characters stacked top-to-bottom are a different layout.)
        _bw, _bh = c['box'][2] - c['box'][0], c['box'][3] - c['box'][1]
        if (abs(ang) <= 4 and _bh >= 1.5 * _bw and len(text) >= 1
                and not any('⺀' <= ch <= '￯' for ch in text)):
            ang = -90.0
        # the detector polygon, in working pixels: the wipe is bounded by this rather than by the box,
        # which for a slanted label covers 1.67x the text area and drags linework into the wipe
        # The polygon is carried whatever QUAD_WIPE says: bounding the wipe to it is still off by
        # default, but _measured_band uses its THICKNESS to cap the text strip (a line crossing the
        # lettering inflates the measured glyph height, and the strip then covers the whole box).
        quad = [[float(p[0]) * S, float(p[1]) * S] for p in quad_src] if quad_src else None
        if erase_only:
            # wiped from the ink layer so no ghost is traced, but no text object: a human types it
            labels.append({'text': '', 'box': box, 'angle': ang, 'erase_only': True, 'quad': quad})
            continue
        # A map's place-name symbol - the small hollow square before "Pixian" - is read by OCR as 口
        # (alone, or glued to the name) and its box swallows it. It was then wiped as lettering and
        # retyped as the glyph 口, which the font draws like 二 or 工, while the name was sized to the
        # symbol-plus-word width. The symbol is graphics: leave it in the drawing, keep it out of the box.
        sq = None
        if abs(ang) <= 4:
            if text in ('口', '□'):
                # a lone 口 in a square box is the symbol itself, also when a line crossing it keeps
                # the outline test from recognising it (three of the five on the geology map)
                bw_, bh_ = box[2] - box[0], box[3] - box[1]
                if (_left_square(src_gray0, box, S, alone=True, gray_all=src_gray_all) is not None
                        or (bh_ > 0 and 0.75 <= bw_ / float(bh_) <= 1.33)):
                    continue
            sq = _left_square(src_gray0, box, S, gray_all=src_gray_all)
            if sq is not None and len(text) > 1:
                text = text.lstrip('口□ ').strip() or text
                box = [min(sq + 2 * S, box[2] - 1), box[1], box[2], box[3]]
        parts = [p.strip() for p in text.split('|') if p.strip()]
        if len(parts) > 1:
            for p, sub in zip(parts, _split_box(ink_nh, box, parts)):
                labels.append({'text': p, 'box': sub, 'angle': 0.0})   # merged rows are horizontal
        else:
            # a slanted OCR box already spans the whole rotated label; growing it sideways swallows
            # neighbouring numerals and fault lines
            grown = box if abs(ang) > 4 else expand_to_text(ink_nh, box, len(text))
            if sq is None and abs(ang) <= 4 and grown[0] < box[0]:
                # the box grew left: the square just outside the OCR box (Xindu) is glyph-shaped
                # enough for expand_to_text to take it in, and it was wiped with the name
                sq = _left_square(src_gray0, grown, S, gray_all=src_gray_all)
            if sq is not None:                       # growing must not take the symbol back in
                grown = [max(grown[0], min(sq + 2 * S, grown[2] - 1))] + list(grown[1:])
            labels.append({'text': text, 'box': grown, 'angle': ang, 'quad': quad})
    # label colour is sampled at SOURCE scale: SR sharpens glyph cores far darker than the original
    # print, while a median over the SR mask is dominated by grey anti-aliased flanks (too pale)
    bgr1 = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    # a fault line crossing the box is pure black and would win the darkest-pixel sample (grey 北祁连
    # came out black). Lines are the ink components that leave the box; glyphs stay inside it.
    n_cc, cc_lab, cc_st, _ = cv2.connectedComponentsWithStats(ink_mask.astype(np.uint8), 8)
    for lab_ in labels:                      # sample glyph colour and height before wiping
        x0, y0, x1, y1 = lab_['box']
        lab_['glyph_h'] = measure_glyph_h(ink_nh, (x0, y0, x1, y1))
        lab_['text_h'] = measure_text_h(src_bgr, [v // S for v in lab_['box']])
        m = 2 * S
        crop = cc_lab[y0:y1 + 1, x0:x1 + 1]
        inside = [i for i in np.unique(crop) if i > 0 and cc_st[i][0] >= x0 - m and cc_st[i][1] >= y0 - m
                  and cc_st[i][0] + cc_st[i][2] <= x1 + 1 + m and cc_st[i][1] + cc_st[i][3] <= y1 + 1 + m]
        glyph = np.isin(crop, inside).astype(np.uint8)
        g1 = cv2.resize(glyph, (max(1, glyph.shape[1] // S), max(1, glyph.shape[0] // S)),
                        interpolation=cv2.INTER_AREA) > 0
        sub = bgr1[y0 // S:y0 // S + g1.shape[0], x0 // S:x0 // S + g1.shape[1]]
        g1 = g1[:sub.shape[0], :sub.shape[1]]
        px = sub[g1 & (sub.min(axis=2) < 200)]
        fg = text_colour(src_bgr, [v / S for v in lab_['box']])
        # Coloured lettering is not in the ink layer, so only the foreground sample can see it. Black
        # lettering is, and the ink sample below reads it darker (on the sharpened SR image) than the
        # source's anti-aliased strokes allow - so neutral labels keep the ink sample when there is one.
        neutral = fg is None or max(fg) - min(fg) < 20
        if fg is not None and not (neutral and len(px)):
            c = np.array(fg[::-1], np.float64)                  # BGR
        elif len(px):
            dark = px[np.argsort(px.astype(np.int32).sum(axis=1))[:max(1, len(px) // 10)]]
            c = dark.mean(axis=0)
        else:
            c = np.array([0, 0, 0])
        lab_['color'] = [int(c[2]), int(c[1]), int(c[0])]     # RGB
    ink_mask0 = ink_mask.copy()          # before wiping: defines what counts as an edge halo
    # Dashed rules are runs of short segments, and a short segment looks exactly like a character stroke,
    # so the wipe eats the dashes crossing a label box. Find them first and keep those pixels. This is
    # the same protection the lineart path applies - the colour path used to miss it entirely, so
    # DASH_PROTECT silently had no effect on every colour build.
    protect = None
    if labels and DASH_PROTECT:
        try:
            import dash_protect

            protect, _dash_info = dash_protect.build_protect_mask(
                ink_gray.shape, dash_protect.find_dashes(ink_gray, bgr),
                confs=('high',), mode='ink', margin=2,
                ink=(dash_protect.ink_layer(ink_gray, bgr) > 0))
        except Exception:  # noqa: BLE001 - optional protection, must never be why the wipe fails
            protect = None
    # A coloured label has no glyphs in the INK layer - only the dark rim of its letters - so wiping
    # its whole text strip there removes nothing but the black linework crossing it (the staircase
    # frame through red "nappe" lost 160 px). Its ink wipe is limited to pixels next to strongly
    # coloured ones; black labels are wiped exactly as before.
    _lab_all = _lab(bgr).astype(np.float32)
    # (the rim of a red letter is dark red, a black line crossing it stays neutral right up to it)
    colour_near = (np.hypot(_lab_all[..., 1] - 128.0, _lab_all[..., 2] - 128.0) > 20)
    # Measure every label's run BEFORE the wipe, not after it. The wipe's own idea of where a slanted
    # label's text lies is an inversion of its axis-aligned box, which is singular near 45 degrees and
    # then fell back to wiping the WHOLE box: the map boundary crossing "E Xiang Qian Fold Belt" was
    # deleted, and that label is left blank, so the drawing lost linework and gained nothing. The
    # measured run gives the strip directly. Costs one extra measuring pass over the labels.
    if MEASURED_TEXT and labels:
        try:
            glyph_mask_colour_best(bgr, labels, S, measure_only=True)
        except Exception:  # noqa: BLE001 - a better wipe bound must never be why a build fails
            pass
    cleaned_gray = (erase_text(ink_gray, labels, protect=protect, colour_near=colour_near)
                    if labels else ink_gray)
    # Coloured lettering printed OVER a black line cuts it: the ink layer never had those pixels, so
    # once the letters are gone the frame line has holes where they stood (red "Longmenshan" on the
    # staircase frame). _heal_crossings bridges collinear stubs only across a path that was dark, so it
    # is shown the occluded view - the label's own-colour glyph pixels count as dark - and restores
    # the line through them at the line's own grey.
    occl = ink_gray.copy()
    col_labels = []
    for lab_ in labels:
        c_ = lab_.get('color')
        if not c_ or max(c_) - min(c_) <= 60:
            continue
        ref = _lab(np.array([[c_[::-1]]], np.uint8)).astype(np.float32)[0, 0]
        bx0, by0, bx1, by1 = (int(v) for v in lab_['box'])
        bx0, by0 = max(bx0, 0), max(by0, 0)
        sl = _lab_all[by0:by1 + 1, bx0:bx1 + 1]
        own = ((np.hypot(sl[..., 1] - 128.0, sl[..., 2] - 128.0) > 30)
               & (np.hypot(sl[..., 1] - ref[1], sl[..., 2] - ref[2]) < 25))
        if own.any():
            line_grey = int(np.median(ink_gray[ink_gray < 200])) if (ink_gray < 200).any() else 0
            occl[by0:by1 + 1, bx0:bx1 + 1][own] = line_grey
            col_labels.append(lab_)
    if col_labels:
        cleaned_gray = _heal_crossings(occl, cleaned_gray, col_labels)
    for lab_ in labels:
        # erase_text derives 'tight' from the glyph core in the INK image. A vertical label printed in
        # colour has no core there - Longquanshan tectonic belt kept the lower half of its run plus the
        # faults beside it, and came out at 7 pt, anchored 100 px low. For a vertical run the OCR box
        # length IS the text length, so a tight box that lost much of it falls back to the box.
        if abs(float(lab_.get('angle') or 0.0)) >= 75 and lab_.get('tight'):
            tx0, ty0, tx1, ty1 = lab_['tight']
            bx0, by0, bx1, by1 = lab_['box']
            if (ty1 - ty0) < 0.8 * (by1 - by0):
                lab_['tight'] = [bx0, by0, bx1, by1]
    del _lab_all
    # The wipe removes linework on purpose. Keeping both sides lets the self-check report the case
    # where a label box deleted more than its own glyphs - damage no later step can undo.
    imwrite(os.path.join(work_dir, 'ink_before_wipe.png'), np.where(ink_mask0 > 0, 0, 255).astype(np.uint8))
    imwrite(os.path.join(work_dir, 'ink_no_text.png'), cleaned_gray)
    for lab_ in labels:
        bx0, by0, bx1, by1 = lab_['box']
        tx0, ty0, tx1, ty1 = lab_['tight']
        # faint grey labels are barely in the ink mask: the "tight" box shrinks to a few pixels and
        # the text came out as a dot (藏, 拆, 兰州). The confirmed box is the better estimate then.
        if (tx1 - tx0) < 0.45 * (bx1 - bx0) or (ty1 - ty0) < 0.35 * (by1 - by0):
            lab_['tight'] = [bx0, by0, bx1, by1]
            lab_['glyph_h'] = 0
        elif abs(lab_.get('angle', 0.0)) <= 4:
            # stacked rows (legend notes) bleed into each other's tight box: stay inside own row
            lab_['tight'] = [tx0, max(ty0, by0 - S), tx1, min(ty1, by1 + S)]
    ink_mask = (cleaned_gray < 128).astype(np.uint8)
    src_gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    blur = blur_factor(src_gray, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), [l['box'] for l in labels])
    for lab_ in labels:
        lab_['bold_px'] = (is_bold(src_gray, lab_['tight'], lab_.get('text_h', 0), blur)
                           if abs(lab_.get('angle', 0.0)) <= 4 else 0.0)

    # Fill colour/line under the labels so wiping text leaves no white hole: the pixels erase_text
    # removed are the pure glyphs (crossing lines were kept), so inpaint just those from neighbours
    # and re-assign the palette on the healed image -> colour blocks become continuous under text.
    bgr_orig = bgr.copy()
    text_mask = ((ink_mask0 > 0) & (ink_mask == 0)).astype(np.uint8)
    split_labels = 0
    foreign = np.zeros(bgr_orig.shape[:2], bool)
    if MEASURED_TEXT or COLOUR_WIPE:
        # Two jobs share one pass over the glyph ink, so the switch has to be split inside it:
        #   measure_only - fills in each label's 'measured' run (length, glyph height, centre) which the
        #       font sizing needs. ON by default; touches no pixels.
        #   the returned mask - widens the WIPE to coloured lettering, which the ink wipe never removed
        #       (so it was re-assigned to a colour layer and traced as ghost lettering inside the patch).
        #       ON by default: without it a blue label on a blue patch is never cleared at all.
        _gm, split_labels = glyph_mask_colour_best(bgr_orig, labels, S, measure_only=not COLOUR_WIPE,
                                                   foreign_out=foreign)
        text_mask |= _gm
    # Also tried: subtracting the ink erase_text deliberately kept, on the argument that the
    # conservative wipe already ruled on those pixels. It gives back 25.5% of the destroyed linework
    # but stops removing 2.4% of the glyph ink, and that 2.4% is not spread evenly - `structure` and
    # `depressionbelt` kept a fifth of their lettering and came back as residue. The colour mask is
    # partly there to cover what erase_text is too cautious to wipe, so it cannot simply defer to it.
    healed_px = 0
    if text_mask.any():
        text_mask = cv2.dilate(text_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * S + 1, 2 * S + 1)))
        bgr = cv2.inpaint(bgr, text_mask, 3, cv2.INPAINT_TELEA)
        bgr, healed_px = heal_colour_crossings(bgr_orig, bgr, text_mask, labels,
                                               glyph_ink=(ink_mask0 > 0) & (ink_mask == 0))
        # Lines of another colour than the label they pass through (the fault along "fault-fold") are
        # wiped with the glyphs - leaving them OUT of the pen let the inpaint smear them into the
        # letter holes as thick red bars - and put back from the original here, after the inpaint.
        foreign &= text_mask > 0
        bgr[foreign] = bgr_orig[foreign]
        healed_px += int(foreign.sum())
        idx = assign_palette(bgr, centers)

    ink_dil = cv2.dilate(ink_mask0, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * S + 3, 2 * S + 3)))
    # thin grey rules (graticule) arrive split over several near-identical clusters and broken up:
    # merge them into one layer, close the gaps, drop the speckle
    grid_mask = np.zeros((h2, w2), np.uint8)
    grid_L = []
    rules, rules_meta = [], None
    for i, (c_, kind) in enumerate(zip(centers, kinds)):
        if kind == 'grid':
            grid_mask |= (idx == i).astype(np.uint8)
            grid_L.append(float(c_[0]))
    # light neutral / pastel SOLID areas (grey bars, table headers, pale flowchart boxes) share the "grid"
    # clusters with thin rules; sent to rule detection they came out as bundles of stray lines. Areas much
    # thicker than a rule become flat fill layers; only the thin remainder is treated as rules.
    fills = []
    label_zone = np.zeros((h2, w2), np.uint8)
    for lab_ in labels:
        x0, y0, x1, y1 = lab_.get('tight', lab_['box'])
        label_zone[max(y0 - S, 0):y1 + S, max(x0 - S, 0):x1 + S] = 1
    thick = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * S + 1, 4 * S + 1))
    for i, kind in enumerate(kinds):
        if kind != 'grid':
            continue
        cl = (idx == i).astype(np.uint8)
        solid = cv2.morphologyEx(cl, cv2.MORPH_OPEN, thick)
        if solid.sum() < 60 * S * S:
            continue
        solid = cv2.dilate(solid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * S + 1, 2 * S + 1))) & cl
        cnts, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(solid)
        cv2.drawContours(filled, cnts, -1, 1, -1)
        holes = filled & (1 - solid)
        n_h, lbl_h, st_h, _ = cv2.connectedComponentsWithStats(holes, 8)
        for j in range(1, n_h):                   # holes left by wiped text inside a filled box
            x, y, w_, h_ = st_h[j][:4]
            if label_zone[y:y + h_, x:x + w_].any():
                solid[y:y + h_, x:x + w_][lbl_h[y:y + h_, x:x + w_] == j] = 1
        solid = clean_mask(solid, 60 * S * S)
        if solid.any():
            fills.append((i, solid))
            grid_mask &= (1 - cv2.dilate(solid, np.ones((3, 3), np.uint8)))
    if grid_mask.any():
        grid_mask &= (1 - ink_dil)      # strip the anti-aliasing halo that hugs the black linework
        for lab_ in labels:             # and the grey ghost left where a label was wiped
            x0, y0, x1, y1 = lab_.get('tight', lab_['box'])
            grid_mask[max(y0 - S, 0):y1 + S, max(x0 - S, 0):x1 + S] = 0
        grid_mask = rebuild_rules(grid_mask, polylines=rules)
        grid_mask &= (1 - (ink_mask0 > 0).astype(np.uint8))
        if rules:
            # rules become real vector lines (not a traced 3 px band): width and colour from the source,
            # so small inputs no longer get graticule lines 1.5x too heavy for the drawing
            probe = np.zeros((h, w), np.uint8)
            for pl in rules:
                pts = (np.array(pl, np.float32).reshape(-1, 2) / S).astype(np.int32)
                cv2.polylines(probe, [pts], False, 1, thickness=1)
            src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
            neutral = np.hypot(src_lab[..., 1] - 128, src_lab[..., 2] - 128) < 12
            vals = src_bgr.min(axis=2)[(probe > 0) & neutral & (src_bgr.min(axis=2) < 245)]
            g = int(np.median(np.sort(vals)[:max(1, len(vals) // 2)])) if len(vals) else 215
            rules_meta = {'polylines': rules, 'width_px': float(S), 'color': [g, g, g]}
    os.makedirs(os.path.join(work_dir, 'layers'), exist_ok=True)
    layers = []
    ink_done = grid_done = False
    cmasks = color_masks(bgr, idx, centers, kinds, ink_dil, min_area)
    text_zone = np.zeros((h2, w2), np.uint8)
    for lab_ in labels:                   # union of the located and the OCR box: the last glyph of a label
        for bx in (lab_.get('tight', lab_['box']), lab_['box']):     # can lie outside the located one
            x0, y0, x1, y1 = bx
            text_zone[max(y0 - S, 0):y1 + S, max(x0 - S, 0):x1 + S] = 1
    # cv2.inpaint knows nothing about where a region ends. A label crossing a patch border (the inset's
    # "Depression" running out of its rectangle, "Chengdu" on the edge of the yellow basin) is filled
    # from whichever side is nearer, so paper was pushed a dozen pixels into the patch: the rectangle
    # came out with a bite and a spur. The region shapes judged BEFORE the wipe are right - that first
    # colour_masks pass already existed and was only used for its union. Inside the wiped area, a
    # pixel no region claims any more goes back to the region that held it before, but only when it
    # connects to what is left of that region: a text hole or a bitten edge does, whereas lettering of
    # a region's own colour standing on paper (the red "nappe structure belt") has nothing left around
    # it and stays wiped. Colours that live almost entirely inside label boxes are lettering, skipped.
    # Measured on the geology map: 2,094 px given back, all on bitten edges and text holes.
    restored_px = 0
    if text_mask.any():
        wiped = text_mask > 0
        taken = np.zeros((h2, w2), bool)
        for m_ in cmasks.values():
            taken |= m_ > 0
        for i_, pre in cmasks_pre.items():
            pre = pre > 0
            if i_ not in cmasks or not pre.any():
                continue
            if float((pre & (text_zone > 0)).sum()) > 0.5 * float(pre.sum()):
                continue
            cur = cmasks[i_] > 0
            back = pre & wiped & ~taken
            if not back.any():
                continue
            _, lbl_ = cv2.connectedComponents((cur | back).astype(np.uint8), connectivity=8)
            keep_ = np.unique(lbl_[cur])
            back &= np.isin(lbl_, keep_[keep_ > 0])
            if back.any():
                cmasks[i_] = (cur | back).astype(np.uint8)
                taken |= back
                restored_px += int(back.sum())
    for i_ in list(cmasks):
        cmasks[i_] = fill_text_holes(cmasks[i_], text_zone, max_hole=int(0.01 * h2 * w2))
    for j_, (i_, fm_) in enumerate(fills):
        fills[j_] = (i_, fill_text_holes(fm_, text_zone, max_hole=int(0.01 * h2 * w2)))
    # The anti-aliased edge of a COLOURED line lands in clusters of its own: dark red, orange, pale red
    # along a red curve. The ink-halo test in color_masks only knows halos of black ink, so each fringe
    # became a layer of thousands of 2 px specks - on a spectrum plot at S=1 (no super-resolution to
    # sharpen the edge) three of them, 4-5.5k specks each, and tracing them hung CorelDRAW until the COM
    # watchdog killed the build. A fringe is mostly dust AND lies along a bigger layer; over 30-odd
    # colour layers of 10 test drawings only those three fringes pass these tests (dust > 0.5, hug > 0.8),
    # the next-dustiest real layer is a faint spectrum line at 0.24.
    fringe_layers = 0
    for i_ in list(cmasks):
        m_ = cmasks[i_] > 0
        tot = int(m_.sum())
        if not tot:
            continue
        _n, _l, st_, _c = cv2.connectedComponentsWithStats(m_.astype(np.uint8), 8)
        a_ = st_[1:, 4]
        # ...and its pieces are specks in the literal sense: the fringes average 1.8-2.6 px. A JPEG'd
        # map's small dark-green patches passed the two tests below just barely (0.54 / 0.81) but
        # average 32 px, and dropping them lost 10 patches on the tibet_jpeg regression case.
        if a_.size == 0 or tot / float(a_.size) >= 4 * S * S:
            continue
        if float(a_[a_ < 20 * S * S].sum()) <= 0.5 * tot:
            continue
        others = ink_mask > 0
        for j_, o_ in cmasks.items():
            if j_ != i_ and int((o_ > 0).sum()) > tot:
                others = others | (o_ > 0)
        near = cv2.dilate(others.astype(np.uint8), np.ones((4 * S + 1, 4 * S + 1), np.uint8)) > 0
        if float((m_ & near).sum()) > 0.8 * tot:
            del cmasks[i_]
            fringe_layers += 1
    # Regions run on UNDER the lines drawn across them. A fault through the pink belt left a hole of
    # its own shape in the pink layer; traced, and rimmed with the patch outline, that hole rendered as
    # a two-edged tube around the fault. Once the fault is a centerline stroke laid on top, the region
    # should simply continue beneath it: close each region across the line pixels, a kernel just wider
    # than a line, so only a gap with the same region on both sides is filled.
    line_px = np.zeros((h2, w2), np.uint8)
    line_ids = [i_ for i_ in list(cmasks) if line_strokes(cmasks[i_], [0, 0, 0]) is not None]
    if line_ids:
        # The thin anti-aliased line itself is patchy in its own layer: part of its pixels went to no
        # layer at all (they showed as white gaps in the tube), so its skeleton came out dashed. Grow
        # each line layer through pixels nobody owns but that lie enclosed by regions - never out
        # into open paper - before it is skeletonised.
        owned = np.zeros((h2, w2), np.uint8)
        for m_ in cmasks.values():
            owned |= (m_ > 0).astype(np.uint8)
        for _i, fm_ in fills:
            owned |= (fm_ > 0).astype(np.uint8)
        owned |= (ink_mask > 0).astype(np.uint8)
        regions = owned.copy()
        for i_ in line_ids:
            regions &= (cmasks[i_] == 0).astype(np.uint8)
        enclosed = (cv2.morphologyEx(regions, cv2.MORPH_CLOSE, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (4 * S + 5, 4 * S + 5))) > 0) & (owned == 0)
        k3 = np.ones((3, 3), np.uint8)
        for i_ in line_ids:
            g_ = (cmasks[i_] > 0).astype(np.uint8)
            for _ in range(6 * S):
                n_ = (cv2.dilate(g_, k3) > 0) & enclosed
                n_ = n_.astype(np.uint8) | g_
                if int(n_.sum()) == int(g_.sum()):
                    break
                g_ = n_
            if line_strokes(g_, [0, 0, 0]) is None:
                g_ = (cmasks[i_] > 0).astype(np.uint8)      # growing must never turn it into a patch
            cmasks[i_] = g_
            line_px |= g_
    if line_px.any():
        line_px = cv2.dilate(line_px, np.ones((3, 3), np.uint8))
        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * S + 5, 4 * S + 5))
        for i_ in list(cmasks):
            r_ = (cmasks[i_] > 0).astype(np.uint8)
            if line_strokes(r_, [0, 0, 0]) is not None:
                continue
            cmasks[i_] = r_ | (cv2.morphologyEx(r_, cv2.MORPH_CLOSE, kc) & line_px)
        for j_, (i_, fm_) in enumerate(fills):
            r_ = (fm_ > 0).astype(np.uint8)
            fills[j_] = (i_, r_ | (cv2.morphologyEx(r_, cv2.MORPH_CLOSE, kc) & line_px))
    all_color = np.zeros((h2, w2), bool)
    for m_ in cmasks.values():
        all_color |= m_ > 0
    ink_mask = drop_specks(drop_patch_outlines(
        ink_mask, all_color, cv2.resize(src_gray, (w2, h2), interpolation=cv2.INTER_NEAREST), text_zone))
    ink_mask, faint_mask, faint_rgb, _ = split_faint_ink(ink_mask, cmasks, centers, src_bgr)
    # linework other layers already draw - a patch outline must not redraw it (see patch_outline)
    drawn_px = (ink_mask > 0) | (faint_mask > 0)
    for i_ in line_ids:
        drawn_px |= cmasks[i_] > 0
    # Text wipe works on the INK image, and only on pixels dark enough to be ink. The lighter anti-aliased
    # rim of grey lettering stays in the colour image, lands in grey clusters (a rim-line layer, the coal
    # band) and is traced into a faint copy of the old text under the new text object. Whatever piece of a
    # layer lies wholly inside a label box is that rim: remove it (the seam fill below closes the hole).
    # A line that crosses the label also runs outside the box and is left alone.
    # Judged against the UNION of the boxes: stacked labels ("岩性" over "油气藏") share one blob of rim
    # that touches each box's edge but never leaves the text area.
    tzone = np.zeros((h2, w2), bool)
    for lab_ in labels:
        bx0, by0, bx1, by1 = lab_.get('box', lab_.get('tight'))
        tzone[max(by0 - S, 0):by1 + S + 1, max(bx0 - S, 0):bx1 + S + 1] = True
    # Colour layers only, and only pieces the LABEL'S OWN INK colour: a legend symbol that happens to sit
    # inside a text box (the map's circle-and-dot marker, a small orange patch) is not text and must stay.
    # The ink layer keeps erase_text's own protections instead.
    # A rim of wiped text is the label's ink BLENDED with what is behind it, so test against that segment,
    # not against the ink colour itself (the rim measured 100 units away from it).
    lab_img = cv2.cvtColor(bgr_orig, cv2.COLOR_BGR2LAB).astype(np.float32)
    tcols = []
    for lab_ in labels:
        c_ = lab_.get('color')
        if not c_:
            continue
        bx0, by0, bx1, by1 = [int(v) for v in (lab_.get('box') or lab_['tight'])]
        ry0, ry1 = max(by0 - 6, 0), min(by1 + 7, h2)
        rx0, rx1 = max(bx0 - 6, 0), min(bx1 + 7, w2)
        ring = np.ones((ry1 - ry0, rx1 - rx0), bool)
        ring[max(by0 - ry0, 0):by1 - ry0 + 1, max(bx0 - rx0, 0):bx1 - rx0 + 1] = False
        px = lab_img[ry0:ry1, rx0:rx1][ring]
        if len(px) < 10:
            continue
        tcols.append((cv2.cvtColor(np.uint8([[list(c_)[::-1]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32),
                      np.median(px, axis=0).astype(np.float32)))
    if tzone.any() and tcols:
        for tgt in [cmasks[i_] for i_ in cmasks] + [fm_ for _i, fm_ in fills]:
            if not (tgt[tzone] > 0).any():
                continue
            n_, lb_ = cv2.connectedComponents((tgt > 0).astype(np.uint8), connectivity=8)
            outside = set(np.unique(lb_[(tgt > 0) & ~tzone]).tolist())
            inside = set(np.unique(lb_[(tgt > 0) & tzone]).tolist()) - {0}
            gone = []
            for q in inside:
                if q in outside:
                    continue
                px = lab_img[lb_ == q]
                if len(px) == 0:
                    continue
                med = np.median(px, axis=0).astype(np.float32)
                for tc, bg in tcols:
                    ab = bg - tc
                    t_ = float(np.clip(np.dot(med - tc, ab) / max(float(np.dot(ab, ab)), 1.0), 0, 1))
                    if float(np.linalg.norm(tc + t_ * ab - med)) < 20:
                        gone.append(q)            # ink of a label blended with its background: a rim
                        break
            if gone:
                tgt[np.isin(lb_, gone)] = 0
    solid_masks = {('c', i_): cmasks[i_] for i_ in cmasks if i_ not in line_ids}
    solid_masks.update({('f', j_): fm_ for j_, (_i, fm_) in enumerate(fills)})
    slivers = drop_boundary_slivers(solid_masks, [ink_mask, faint_mask] + [cmasks[i_] for i_ in line_ids],
                                     cv2.resize(bgr_orig, (w2, h2), interpolation=cv2.INTER_NEAREST))
    # Underpaint. Layers are traced one at a time and painted big-first, so two neighbouring regions meet
    # along two independently traced edges that never coincide: paper shows through as a white hairline
    # along every band. Each region is therefore extended under the holes that the regions painted on
    # top of it fill - the seam then shows the colour beneath, not paper. Holes nothing covers (a white
    # island in the drawing) stay holes.
    order_ = sorted(solid_masks, key=lambda k_: -int((solid_masks[k_] > 0).sum()))
    above = np.zeros((h2, w2), bool)
    for m_ in [ink_mask, faint_mask] + [cmasks[i_] for i_ in line_ids]:
        above |= m_ > 0
    tops = {}
    acc = above.copy()
    for k_ in reversed(order_):                   # smallest first: what lies above each layer
        tops[k_] = acc.copy()
        acc |= solid_masks[k_] > 0
    kd = np.ones((2 * S + 1, 2 * S + 1), np.uint8)
    for k_ in order_:
        m_ = (solid_masks[k_] > 0).astype(np.uint8)
        if not m_.any():
            continue
        cnts_, hier_ = cv2.findContours(m_, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier_ is None:
            continue
        cover = cv2.dilate(tops[k_].astype(np.uint8), kd) > 0
        for q, hq in enumerate(hier_[0]):
            if hq[3] < 0:
                continue                          # outer contour
            hole = np.zeros_like(m_)
            cv2.drawContours(hole, cnts_, q, 1, -1)
            hole = (hole > 0) & (m_ == 0)
            hn = int(hole.sum())
            if hn and float((hole & cover).sum()) >= 0.95 * hn:
                m_[hole] = 1
        solid_masks[k_] = m_
    for (src_, j_), m_ in solid_masks.items():
        if src_ == 'c':
            cmasks[j_] = m_
        else:
            fills[j_] = (fills[j_][0], m_)
    for i, (c, kind) in enumerate(zip(centers, kinds)):
        if kind == 'grid':
            if grid_done or not grid_mask.any() or rules_meta:
                continue
            grid_done = True
            mask = grid_mask
            # the ~1 px rules were widened to 3 px: spread the same darkness over 3x the width
            c = np.array([255 - (255 - float(np.median(grid_L))) / 3.0, 128, 128], np.float32)
        elif kind == 'ink':
            if ink_done:
                continue          # all dark neutral clusters share one merged ink layer
            ink_done = True
            mask = ink_mask
            c = np.array([min(float(cc[0]) for cc in centers if classify(cc) == 'ink'), 128, 128], np.float32)
        else:
            if i not in cmasks:
                continue
            mask = cmasks[i]
        area = int(mask.sum())
        if not big_enough(mask, min_area, h2 * w2):
            continue
        rgb = interior_color(bgr_orig, mask) if kind == 'color' else lab_to_bgr(c)[::-1]
        f = 'layers/layer_%02d_%s.png' % (i, kind)
        imwrite(os.path.join(work_dir, f), np.where(mask > 0, 0, 255).astype(np.uint8))
        entry = {'file': f, 'kind': kind, 'area': area, 'color': [int(rgb[0]), int(rgb[1]), int(rgb[2])]}
        strokes = line_strokes(mask, entry['color']) if kind == 'color' else None
        rest = None
        if not strokes and kind == 'color':
            dashed = dashed_strokes(mask, entry['color'])
            if dashed:
                strokes, rest = dashed
        if strokes:
            entry['strokes'] = strokes
            if rest is not None and int(rest.sum()) >= min_area:
                # what the dash chains do not explain is still traced, so nothing of the layer is lost
                fr = 'layers/layer_%02d_rest.png' % i
                imwrite(os.path.join(work_dir, fr), np.where(rest > 0, 0, 255).astype(np.uint8))
                entry['rest_file'] = fr
        elif kind == 'color':
            rim = patch_outline(src_bgr, mask, entry['color'], drawn=drawn_px)
            if rim:
                entry['outline'] = {'color': rim, 'width_px': float(S)}      # ~1 source px
        layers.append(entry)
    for i, fm in fills:
        if not fm.any():
            continue                              # emptied: it was only the rim of wiped text
        f = 'layers/layer_%02d_fill.png' % i
        imwrite(os.path.join(work_dir, f), np.where(fm > 0, 0, 255).astype(np.uint8))
        layers.append({'file': f, 'kind': 'color', 'area': int(fm.sum()), 'color': interior_color(bgr_orig, fm)})
    if faint_rgb is not None and faint_mask.sum() >= min_area:
        f = 'layers/layer_99_ink_faint.png'
        imwrite(os.path.join(work_dir, f), np.where(faint_mask > 0, 0, 255).astype(np.uint8))
        layers.append({'file': f, 'kind': 'ink_faint', 'area': int(faint_mask.sum()), 'color': faint_rgb})
    # paint order: big colour regions first, smaller ones on top, ink last
    order = {'color': 0, 'grid': 1, 'ink_faint': 2, 'ink': 3}
    layers.sort(key=lambda l: (order.get(l['kind'], 0), -l['area']))
    for l in labels:
        ang = abs(float(l.get('angle') or 0.0))
        # A slanted label with no trustworthy measurement cannot be sized from its box either - that
        # inversion is singular near 45 degrees and its fallback lets the box decide the font size, which
        # is the giant-label defect. Leaving it blank is the lesser evil and follows the measured switch.
        l['unmeasurable'] = bool(MEASURED_TEXT and 4.0 < ang < 75.0 and not l.get('measured'))
    out_labels = [{'text': l['text'], 'tight': l['tight'], 'box': l['box'], 'color': l['color'],
                   'glyph_h': l.get('glyph_h', 0), 'text_h': l.get('text_h', 0), 'bold_px': l.get('bold_px', 0.0),
                   'angle': l.get('angle', 0.0), 'erase_only': bool(l.get('erase_only')),
                   'unmeasurable': bool(l.get('unmeasurable')), 'measured': l.get('measured'),
                   'quad': l.get('quad'), 'run_gh': l.get('run_gh'),
                   # wipe_protected only ever counts the PROTECT_SHAPE branch, which is off by default,
                   # so it stayed 0 whatever the dash protection did and was read as "protection never
                   # ran". dash_protect_px is the counter that actually moves, and wipe_colour_px is
                   # the size of the colour wipe - an inpaint that wipe_spill is structurally unable
                   # to see, so without it that erase has no observable at all.
                   'dash_protect_px': int(l.get('dash_protect_px') or 0),
                   'wipe_colour_px': int(l.get('wipe_colour_px') or 0),
                   'wipe_protected': int(l.get('wipe_protected') or 0)} for l in labels]
    meta = {'src_size': [w, h], 'scale': S, 'layers': layers, 'rules': rules_meta,
            'latin_font': os.environ.get('CDR_LATIN_FONT') or detect_latin_font(src_gray, labels),
            'wipe_protected_px': int(sum(int(l.get('wipe_protected') or 0) for l in labels)),
            'dash_protect_px': int(sum(int(l.get('dash_protect_px') or 0) for l in labels)),
            'wipe_colour_px': int(sum(int(l.get('wipe_colour_px') or 0) for l in labels)),
            'colour_healed_px': int(healed_px),
            'colour_restored_px': int(restored_px),
            'labels_hue_split': int(split_labels),
            'fringe_layers_dropped': int(fringe_layers),
            'boundary_slivers_dropped': int(slivers),
            'labels': annotate_placement(out_labels, S)}
    with open(os.path.join(work_dir, 'layers.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    return meta



def maybe_flat(work_dir, meta):
    """Switch a flat-colour drawing to the region model (cdr_flat): CDR_FLAT=1 forces it, 0 disables it,
    unset decides by flat_score(). The labels measured by build_layers() are kept either way."""
    flag = os.environ.get('CDR_FLAT', '').strip()
    if flag == '0':
        return meta
    import cdr_flat
    src = imread(os.path.join(work_dir, 'src.png'), cv2.IMREAD_COLOR)
    ok, stats = cdr_flat.flat_score(src)
    meta['flat_score'] = stats
    if flag != '1' and not ok:
        return meta
    try:
        res = cdr_flat.build_flat(work_dir, meta, S=int(meta.get('scale') or S), log=lambda *a: None)
    except Exception as exc:  # noqa: BLE001 - the layer model is still there to fall back on
        meta['flat_error'] = repr(exc)[:300]
        return meta
    if res.get('flat') is False:
        return meta
    with open(os.path.join(work_dir, 'layers.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    return meta

if __name__ == '__main__':
    if sys.argv[1] == 'layers':
        m = build_layers(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 6)
        maybe_flat(sys.argv[2], m)
        print(json.dumps({'layers': [{k: v for k, v in l.items() if k not in ('file', 'shapes', 'strokes')}
                                     for l in m['layers']],
                          'flat': m.get('flat'), 'labels': len(m['labels'])}, ensure_ascii=False))
    else:
        raise SystemExit(__doc__)
