"""Smooth contour tracing for image -> CorelDRAW conversion.

Why this exists (both earlier attempts failed in opposite ways):

  * contour tracing without smoothing  -> keeps glyphs and solid marks intact,
    but the outlines follow the raw pixel staircase, so the result looks fuzzy.
  * skeleton centerline (stroked)       -> crisp strokes, but CJK glyphs become
    unreadable single-line skeletons and solid areas collapse into outlines.

This module keeps the FILLED-contour representation (so text stays readable and
solid marks stay solid) and removes the staircase by low-pass filtering the
contour points, with gentle smoothing on small shapes (glyphs deform easily)
and stronger smoothing on long ones (lines, borders).

Output SVG: black outer contours + white hole contours, both smooth.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _smooth_closed(pts: np.ndarray, iterations: int, window: int = 5) -> np.ndarray:
    """Low-pass filter a CLOSED contour to remove pixel staircase."""
    if len(pts) < window + 1 or iterations <= 0:
        return pts
    half = window // 2
    out = pts.astype(np.float64)
    n = len(out)
    for _ in range(iterations):
        idx = np.arange(n)
        acc = np.zeros_like(out)
        for k in range(-half, half + 1):
            acc += out[(idx + k) % n]
        out = acc / (2 * half + 1)
    return out


def _dedupe(pts: np.ndarray, min_dist: float = 1.2) -> np.ndarray:
    """Drop points that sit almost on top of the previous one."""
    if len(pts) < 2:
        return pts
    keep = [pts[0]]
    for p in pts[1:]:
        if np.hypot(*(p - keep[-1])) >= min_dist:
            keep.append(p)
    if len(keep) >= 3 and np.hypot(*(keep[0] - keep[-1])) < min_dist:
        keep.pop()
    return np.array(keep)


def vectorize_smooth(
    src: str,
    out_svg: str,
    thresh: int = 175,
    blur: int = 3,
    min_area: float = 1.0,
    detail: float = 0.5,
    smooth_small: int = 0,
    smooth_large: int = 2,
    small_size: int = 24,
    invert: bool = False,
) -> dict:
    """Vectorize `src` into smooth filled contours (holes preserved).

    Defaults are tuned by measurement, not taste: glyph-bearing shapes must NOT
    be smoothed (smoothing merges adjacent strokes and destroys CJK glyphs),
    while large shapes get two low-pass passes to remove the pixel staircase.
    detail=0.5 keeps thin strokes; raising it erodes glyph detail.
    """
    from cdr_check import imread_unicode

    gray = imread_unicode(src, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(src)
    h, w = gray.shape[:2]

    if blur and blur > 1:
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)
    mode = cv2.THRESH_BINARY if invert else cv2.THRESH_BINARY_INV
    _, mask = cv2.threshold(gray, thresh, 255, mode)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = hierarchy[0] if hierarchy is not None else []

    black: list[np.ndarray] = []
    white: list[np.ndarray] = []
    pts_total = 0

    for idx, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < min_area:
            continue
        ap = cv2.approxPolyDP(cnt, detail, True).reshape(-1, 2).astype(np.float64)
        if len(ap) < 3:
            continue
        # small shapes are glyphs/strokes: smooth lightly so they stay legible
        iters = smooth_small if len(ap) < small_size else smooth_large
        sm = _smooth_closed(ap, iters)
        sm = _dedupe(sm)
        if len(sm) < 3:
            continue
        pts_total += len(sm)
        is_hole = idx < len(hierarchy) and hierarchy[idx][3] != -1
        (white if is_hole else black).append(sm)

    def path_of(pts: np.ndarray) -> str:
        d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + " Z"
        return f'<path d="{d}"/>'

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">',
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>',
        '<g fill="#000000" stroke="none">',
    ]
    svg += [path_of(p) for p in black]
    svg.append("</g>")
    if white:
        svg.append('<g fill="#ffffff" stroke="none">')
        svg += [path_of(p) for p in white]
        svg.append("</g>")
    svg.append("</svg>")
    with open(out_svg, "w", encoding="utf-8") as fh:
        fh.write("\n".join(svg))

    return {
        "size": [w, h],
        "outer_contours": len(black),
        "hole_contours": len(white),
        "points": pts_total,
        "svg": out_svg,
    }


if __name__ == "__main__":
    src = sys.argv[1]
    out = sys.argv[2]
    info = vectorize_smooth(src, out)
    for k, v in info.items():
        print(f"{k}: {v}")
