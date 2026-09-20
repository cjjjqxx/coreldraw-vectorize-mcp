"""Skeleton-based vectorization: emit sharp, editable vector STROKES.

The previous approach (contour tracing -> filled polygons) faithfully copied
every anti-aliased pixel edge, so the output stayed as fuzzy and jagged as the
source and was not editable linework.

This module instead:
  1. cleans the raster (blur, threshold, drop speckles),
  2. reduces linework to a 1-pixel CENTERLINE (Zhang-Suen thinning),
  3. walks the skeleton into polyline paths (junction/endpoint graph),
  4. simplifies each path and emits SVG <path> with a stroke and no fill.

CorelDRAW imports those as curve objects with an outline width, so the result
is crisp at any zoom and genuinely editable (node editing, line weight,
colour) - which is the actual point of "image -> CDR".
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def preprocess(
    gray: np.ndarray, thresh: int = 170, blur: int = 3, min_area: int = 4
) -> np.ndarray:
    """Binary linework mask with speckles removed."""
    if blur and blur > 1:
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return keep


def skeletonize(mask: np.ndarray) -> np.ndarray:
    """1-pixel wide centerline via Zhang-Suen thinning.

    opencv-contrib's cv2.ximgproc.thinning when present; plain opencv-python (what requirements.txt
    installs, and OpenCV 5 has no thinning at all) gets the same algorithm in numpy below.
    """
    thinning = getattr(getattr(cv2, 'ximgproc', None), 'thinning', None)
    if thinning is not None:
        return (thinning(mask) > 0).astype(np.uint8)
    return _zhang_suen((mask > 0).astype(np.uint8))


def _zhang_suen(img: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning (1984), vectorised; works on the mask's bounding box only."""
    ys, xs = np.nonzero(img)
    out = np.zeros_like(img, np.uint8)
    if ys.size == 0:
        return out
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    a = np.pad(img[y0:y1, x0:x1].astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            p2, p3, p4 = a[:-2, 1:-1], a[:-2, 2:], a[1:-1, 2:]
            p5, p6, p7 = a[2:, 2:], a[2:, 1:-1], a[2:, :-2]
            p8, p9 = a[1:-1, :-2], a[:-2, :-2]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            b = sum(n.astype(np.int16) for n in nb)
            seq = nb + [p2]
            t = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                c1, c2 = p2 * p4 * p6, p4 * p6 * p8
            else:
                c1, c2 = p2 * p4 * p8, p2 * p6 * p8
            kill = (a[1:-1, 1:-1] == 1) & (b >= 2) & (b <= 6) & (t == 1) & (c1 == 0) & (c2 == 0)
            if kill.any():
                a[1:-1, 1:-1][kill] = 0
                changed = True
        if not changed:
            break
    out[y0:y1, x0:x1] = a[1:-1, 1:-1]
    return out


_N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def trace_paths(skel: np.ndarray) -> list[list[tuple[int, int]]]:
    """Walk the skeleton into polyline paths.

    Nodes (degree != 2) split the skeleton into segments; pure loops left over
    are extracted as closed paths.
    """
    pts = set(zip(*np.nonzero(skel)))  # (y, x)

    def nbrs(p):
        # m-adjacency: a diagonal neighbour counts only when neither shared 4-neighbour is set. With
        # plain 8-adjacency every stair step of a thinned diagonal is a little triangle, each of its
        # pixels has degree 3, and the walk shattered one fault line into dozens of 1-2 px paths.
        y, x = p
        out = []
        for dy, dx in _N8:
            q = (y + dy, x + dx)
            if q not in pts:
                continue
            if dy and dx and ((y + dy, x) in pts or (y, x + dx) in pts):
                continue
            out.append(q)
        return out

    deg = {p: len(nbrs(p)) for p in pts}
    nodes = {p for p in pts if deg[p] != 2}

    used: set[frozenset] = set()
    paths: list[list[tuple[int, int]]] = []

    for node in nodes:
        for nb in nbrs(node):
            e = frozenset((node, nb))
            if e in used:
                continue
            used.add(e)
            path = [node, nb]
            prev, cur = node, nb
            while cur not in nodes:
                nxt = [q for q in nbrs(cur) if q != prev]
                if not nxt:
                    break
                nn = nxt[0]
                used.add(frozenset((cur, nn)))
                path.append(nn)
                prev, cur = cur, nn
            if len(path) >= 2:
                paths.append(path)

    # leftover loops: every remaining pixel has degree 2
    consumed = {p for path in paths for p in path}
    remaining = pts - consumed
    while remaining:
        start = next(iter(remaining))
        loop = [start]
        prev, cur = None, start
        while True:
            cand = [q for q in nbrs(cur) if q != prev and q in remaining]
            if not cand:
                break
            nn = cand[0]
            loop.append(nn)
            remaining.discard(cur)
            prev, cur = cur, nn
            if cur == start:
                break
        remaining.discard(start)
        if len(loop) >= 3:
            loop.append(start)
            paths.append(loop)

    return paths


def _simplify(path: list[tuple[int, int]], eps: float) -> np.ndarray:
    arr = np.array([[x, y] for y, x in path], dtype=np.int32).reshape(-1, 1, 2)
    if len(path) <= 2 or eps <= 0:
        return arr
    return cv2.approxPolyDP(arr, eps, False)


def estimate_stroke_width(mask: np.ndarray, skel: np.ndarray) -> float:
    """Average line width = linework area / centerline length."""
    area = float(np.count_nonzero(mask))
    length = float(np.count_nonzero(skel))
    if length <= 0:
        return 1.0
    return max(0.8, area / length)


def vectorize_strokes(
    src: str,
    out_svg: str,
    thresh: int = 170,
    blur: int = 3,
    min_area: int = 4,
    min_len: int = 4,
    simplify: float = 1.0,
    smooth: bool = True,
    width_scale: float = 1.0,
    stroke_width: float | None = None,
) -> dict:
    """Vectorize `src` into stroked SVG paths. Returns statistics."""
    from cdr_check import imread_unicode

    gray = imread_unicode(src, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(src)
    h, w = gray.shape[:2]

    mask = preprocess(gray, thresh=thresh, blur=blur, min_area=min_area)
    skel = skeletonize(mask)
    raw = trace_paths(skel)

    sw = stroke_width if stroke_width is not None else estimate_stroke_width(mask, skel)
    sw *= width_scale

    kept = []
    for path in raw:
        if len(path) < min_len:
            continue
        ap = _simplify(path, simplify)
        if len(ap) < 2:
            continue
        kept.append(ap)

    # optional smoothing: soften staircase artefacts of the pixel skeleton
    if smooth and len(kept) > 1:
        kept = [_chaikin(ap, 1) for ap in kept]

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">',
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
        f'<g fill="none" stroke="#000000" stroke-width="{sw:.2f}" '
        'stroke-linecap="round" stroke-linejoin="round">',
    ]
    pts_total = 0
    for ap in kept:
        coords = ap.reshape(-1, 2)
        pts_total += len(coords)
        d = "M " + " L ".join(f"{int(x)},{int(y)}" for x, y in coords)
        svg.append(f'<path d="{d}"/>')
    svg += ["</g>", "</svg>"]
    with open(out_svg, "w", encoding="utf-8") as fh:
        fh.write("\n".join(svg))

    return {
        "size": [w, h],
        "stroke_width": round(sw, 2),
        "skeleton_px": int(np.count_nonzero(skel)),
        "mask_px": int(np.count_nonzero(mask)),
        "raw_paths": len(raw),
        "paths": len(kept),
        "points": pts_total,
        "svg": out_svg,
    }


def _chaikin(ap: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Chaikin corner cutting: turns jagged polyline corners into smooth ones."""
    pts = ap.reshape(-1, 2).astype(np.float64)
    if len(pts) < 3:
        return ap
    for _ in range(iterations):
        out = [pts[0]]
        for i in range(len(pts) - 1):
            p, q = pts[i], pts[i + 1]
            out.append(0.75 * p + 0.25 * q)
            out.append(0.25 * p + 0.75 * q)
        out.append(pts[-1])
        pts = np.array(out)
    return pts.astype(np.int32).reshape(-1, 1, 2)


if __name__ == "__main__":
    src = sys.argv[1]
    out = sys.argv[2]
    info = vectorize_strokes(src, out)
    for k, v in info.items():
        print(f"{k}: {v}")
