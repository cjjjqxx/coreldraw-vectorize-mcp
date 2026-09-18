"""Vectorize a line-art image into SVG paths (real tracing, not hand drawing).

Pipeline: grayscale -> threshold -> denoise -> findContours -> approxPolyDP
simplify -> SVG path elements. The result reproduces the reference geometry
instead of approximating it with hand-placed primitives.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cdr_check import imread_unicode  # noqa: E402


def vectorize(
    src: str,
    out_svg: str,
    thresh: int = 170,
    simplify: float = 1.2,
    min_area: float = 1.5,
    blur: int = 3,
) -> dict:
    img = imread_unicode(src, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(src)
    h, w = img.shape[:2]

    if blur and blur > 1:
        img = cv2.GaussianBlur(img, (blur, blur), 0)

    # linework = dark pixels
    _, mask = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY_INV)
    # remove single-pixel noise
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    )

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0] if hierarchy is not None else []

    # Split outer contours from holes. CorelDRAW does not honour
    # fill-rule="evenodd" on SVG import: it fills every path solid, which turns
    # text and thin strokes into black blobs. Drawing holes as white overlays
    # on top of the black fills reproduces the intended shape everywhere.
    outer_paths: list[str] = []
    hole_paths: list[str] = []
    pts_total = 0
    for idx, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < min_area:
            continue
        ap = cv2.approxPolyDP(cnt, simplify, True)
        if len(ap) < 3:
            continue
        pts = ap.reshape(-1, 2)
        pts_total += len(pts)
        d = "M " + " L ".join(f"{int(x)},{int(y)}" for x, y in pts) + " Z"
        is_hole = idx < len(hierarchy) and hierarchy[idx][3] != -1
        (hole_paths if is_hole else outer_paths).append(d)

    w_, h_ = w, h
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w_}" height="{h_}" '
        f'viewBox="0 0 {w_} {h_}">',
        f'<rect width="{w_}" height="{h_}" fill="#ffffff"/>',
        '<g fill="#000000" stroke="none">',
    ]
    svg += [f'<path d="{d}"/>' for d in outer_paths]
    svg += ["</g>"]
    if hole_paths:
        svg += ['<g fill="#ffffff" stroke="none">']
        svg += [f'<path d="{d}"/>' for d in hole_paths]
        svg += ["</g>"]
    svg += ["</svg>"]
    with open(out_svg, "w", encoding="utf-8") as fh:
        fh.write("\n".join(svg))

    return {
        "size": [w, h],
        "contours_found": len(contours),
        "paths_kept": len(outer_paths) + len(hole_paths),
        "outer_paths": len(outer_paths),
        "hole_paths": len(hole_paths),
        "points_total": pts_total,
        "svg": out_svg,
    }


if __name__ == "__main__":
    src = sys.argv[1]
    out = sys.argv[2]
    info = vectorize(src, out)
    for k, v in info.items():
        print(f"{k}: {v}")
