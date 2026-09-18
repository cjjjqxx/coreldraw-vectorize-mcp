"""Colour layering for the vectorize pipeline.

A colour drawing is split into flat layers instead of letting PowerTRACE quantise colour itself:
the image is reduced to a small palette, each palette colour becomes one black-on-white mask that
is traced separately and filled with that colour, and the dark neutral "ink" (lines + text) becomes
the top layer. This keeps region edges clean and keeps the linework from being broken up by colour.

CLI: python cdr_vectorize_color.py layers <work_dir> <labels_confirmed.json> [colors]
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdr_vectorize import (S, erase_text, expand_to_text, imread, imwrite, _line_mask, _rule_mask,  # noqa: E402
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


def drop_patch_outlines(ink_mask, color_px):
    """Filled patches carry a 1-2 px darker outline that lands in the ink layer as long thin rims
    (drop_region_rims only catches short crumbs). Remove thin ink (no 5 px wide core) whose
    connected piece mostly hugs a colour patch; fault lines are thick and kept, and a small map
    symbol that merely touches a patch is only partly near it, so it survives."""
    ink = (ink_mask > 0).astype(np.uint8)
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


def patch_outline(src_bgr, mask, fill_rgb):
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
    for m_ in color_masks(bgr, idx, centers, kinds, ink_dil0, min_area).values():
        color_px |= m_ > 0
    ink_mask = drop_region_rims(ink_mask, color_px)

    # labels live in the ink layer: erase them there, then re-create them as text objects
    if isinstance(confirmed, str):
        confirmed = json.load(open(confirmed, encoding='utf-8'))
    cand_path = os.path.join(work_dir, 'ocr_candidates.json')
    cands = json.load(open(cand_path, encoding='utf-8')).get('candidates', []) if os.path.exists(cand_path) else []

    def _angle_for(box_src):
        """angle for a confirmed label: from its own field, else nearest OCR candidate by IoU."""
        best, bi = 0.0, 0.0
        for cd in cands:
            b = cd['box']
            ix = max(0, min(box_src[2], b[2]) - max(box_src[0], b[0]))
            iy = max(0, min(box_src[3], b[3]) - max(box_src[1], b[1]))
            inter = ix * iy
            u = (box_src[2]-box_src[0])*(box_src[3]-box_src[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9
            if inter/u > bi:
                bi, best = inter/u, float(cd.get('angle', 0.0))
        return best if bi > 0.2 else 0.0

    ink_gray = np.where(ink_mask > 0, 0, 255).astype(np.uint8)
    ink_nh = ink_mask & (1 - _rule_mask(ink_mask))
    labels = []
    for c in confirmed or []:
        text = str(c['text']).strip()
        if not text:
            continue
        box = [int(v) * S for v in c['box']]
        box = [max(box[0], 0), max(box[1], 0), min(box[2], w2 - 1), min(box[3], h2 - 1)]
        ang = float(c['angle']) if 'angle' in c else _angle_for([int(v) for v in c['box']])
        parts = [p.strip() for p in text.split('|') if p.strip()]
        if len(parts) > 1:
            for p, sub in zip(parts, _split_box(ink_nh, box, parts)):
                labels.append({'text': p, 'box': sub, 'angle': 0.0})   # merged rows are horizontal
        else:
            # a slanted OCR box already spans the whole rotated label; growing it sideways swallows
            # neighbouring numerals and fault lines
            grown = box if abs(ang) > 4 else expand_to_text(ink_nh, box, len(text))
            labels.append({'text': text, 'box': grown, 'angle': ang})
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
        if len(px):
            dark = px[np.argsort(px.astype(np.int32).sum(axis=1))[:max(1, len(px) // 10)]]
            c = dark.mean(axis=0)
        else:
            c = np.array([0, 0, 0])
        lab_['color'] = [int(c[2]), int(c[1]), int(c[0])]     # RGB
    ink_mask0 = ink_mask.copy()          # before wiping: defines what counts as an edge halo
    cleaned_gray = erase_text(ink_gray, labels) if labels else ink_gray
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
    if text_mask.any():
        text_mask = cv2.dilate(text_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * S + 1, 2 * S + 1)))
        bgr = cv2.inpaint(bgr, text_mask, 3, cv2.INPAINT_TELEA)
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
    for i_ in list(cmasks):
        cmasks[i_] = fill_text_holes(cmasks[i_], text_zone, max_hole=int(0.01 * h2 * w2))
    for j_, (i_, fm_) in enumerate(fills):
        fills[j_] = (i_, fill_text_holes(fm_, text_zone, max_hole=int(0.01 * h2 * w2)))
    all_color = np.zeros((h2, w2), bool)
    for m_ in cmasks.values():
        all_color |= m_ > 0
    ink_mask = drop_specks(drop_patch_outlines(ink_mask, all_color))
    ink_mask, faint_mask, faint_rgb, _ = split_faint_ink(ink_mask, cmasks, centers, src_bgr)
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
        if kind == 'color':
            rim = patch_outline(src_bgr, mask, entry['color'])
            if rim:
                entry['outline'] = {'color': rim, 'width_px': float(S)}      # ~1 source px
        layers.append(entry)
    for i, fm in fills:
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
    meta = {'src_size': [w, h], 'scale': S, 'layers': layers, 'rules': rules_meta,
            'latin_font': detect_latin_font(src_gray, labels),
            'labels': [{'text': l['text'], 'tight': l['tight'], 'box': l['box'], 'color': l['color'],
                        'glyph_h': l.get('glyph_h', 0), 'text_h': l.get('text_h', 0), 'bold_px': l.get('bold_px', 0.0),
                        'angle': l.get('angle', 0.0)} for l in labels]}
    with open(os.path.join(work_dir, 'layers.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    return meta


if __name__ == '__main__':
    if sys.argv[1] == 'layers':
        m = build_layers(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 6)
        print(json.dumps({'layers': [{k: v for k, v in l.items() if k != 'file'} for l in m['layers']],
                          'labels': len(m['labels'])}, ensure_ascii=False))
    else:
        raise SystemExit(__doc__)
