"""Quantitative comparison between a reference drawing and a reproduction.

This is the "check" half of the draw-check-correct loop: it aligns the two
images, extracts linework, and reports coverage/precision plus a per-cell
difference map, so a reproduction can be judged by numbers instead of by
impression.

Metrics (all computed with a pixel tolerance so slight offsets are forgiven):
  recall    - fraction of REFERENCE linework that the reproduction covers
              ("did we draw everything the target has?")
  precision - fraction of REPRODUCTION linework that exists in the reference
              ("did we invent things the target does not have?")
  f1        - harmonic mean of the two
"""
from __future__ import annotations

import cv2
import numpy as np


def imread_unicode(path: str, flags: int = cv2.IMREAD_GRAYSCALE) -> np.ndarray | None:
    """cv2.imread cannot handle non-ASCII (e.g. Chinese) paths on Windows;
    read the bytes ourselves and decode in memory."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """Counterpart to imread_unicode for non-ASCII output paths."""
    ext = "." + path.rsplit(".", 1)[-1]
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True


def _load_linework(path: str, size: tuple[int, int], thresh: int = 160) -> np.ndarray:
    """Load an image as a binary linework mask at the given (w, h).

    Binarize FIRST at the image's native resolution, then rescale the mask.
    Thresholding after a downscale would blur thin strokes into grey and
    wrongly count them as thick linework, which corrupts the precision score.
    """
    img = imread_unicode(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    _, mask = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY_INV)
    if (mask.shape[1], mask.shape[0]) != size:
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return mask


def _dilate(mask: np.ndarray, tol: int) -> np.ndarray:
    if tol <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    return cv2.dilate(mask, k)


def compare(
    reference_path: str,
    candidate_path: str,
    tolerance: int = 4,
    grid: tuple[int, int] = (12, 9),
    thresh: int = 160,
) -> dict:
    """Compare candidate against reference. Returns metrics + grid breakdown."""
    ref0 = imread_unicode(reference_path, cv2.IMREAD_GRAYSCALE)
    if ref0 is None:
        raise FileNotFoundError(reference_path)
    h, w = ref0.shape[:2]
    size = (w, h)

    ref = _load_linework(reference_path, size, thresh) > 0
    cand = _load_linework(candidate_path, size, thresh) > 0

    ref_d = _dilate(ref.astype(np.uint8) * 255, tolerance) > 0
    cand_d = _dilate(cand.astype(np.uint8) * 255, tolerance) > 0

    ref_n = int(ref.sum())
    cand_n = int(cand.sum())
    covered = int((ref & cand_d).sum())      # reference linework matched
    exact = int((cand & ref_d).sum())        # candidate linework that is valid

    recall = covered / ref_n if ref_n else 0.0
    precision = exact / cand_n if cand_n else 0.0
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) else 0.0

    # per-cell recall / precision
    gx, gy = grid
    cw, ch = max(1, w // gx), max(1, h // gy)
    cells = []
    for j in range(gy):
        row = []
        for i in range(gx):
            y0, y1 = j * ch, min(h, (j + 1) * ch)
            x0, x1 = i * cw, min(w, (i + 1) * cw)
            r_n = int(ref[y0:y1, x0:x1].sum())
            c_n = int(cand[y0:y1, x0:x1].sum())
            r_cov = int((ref[y0:y1, x0:x1] & cand_d[y0:y1, x0:x1]).sum())
            c_ok = int((cand[y0:y1, x0:x1] & ref_d[y0:y1, x0:x1]).sum())
            row.append({
                "ref_px": r_n,
                "cand_px": c_n,
                "recall": round(r_cov / r_n, 3) if r_n else None,
                "precision": round(c_ok / c_n, 3) if c_n else None,
            })
        cells.append(row)

    return {
        "reference_size": [w, h],
        "reference_px": ref_n,
        "candidate_px": cand_n,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "grid": {"cols": gx, "rows": gy, "cells": cells},
    }


def ascii_report(result: dict) -> str:
    """Render the grid as a compact ASCII map of per-cell recall."""
    cells = result["grid"]["cells"]
    lines = ["per-cell RECALL (missing reference linework):"]
    header = "      " + "".join(f"{i:^6}" for i in range(result["grid"]["cols"]))
    lines.append(header)
    for j, row in enumerate(cells):
        parts = []
        for c in row:
            r = c["recall"]
            if r is None:
                parts.append("  --  ")
            elif r >= 0.85:
                parts.append("  ok  ")
            elif r >= 0.6:
                parts.append(f" {r:.2f} ")
            elif r >= 0.3:
                parts.append(f"[{r:.2f}]")
            else:
                parts.append(f"<{r:.2f}>")
        lines.append(f" r{j:<3}" + "".join(parts))
    lines.append("legend: ok>=0.85  [ ]=0.3-0.6  <>=below 0.3 (badly missing)")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    ref = sys.argv[1]
    cand = sys.argv[2]
    res = compare(ref, cand)
    print(f"recall={res['recall']} precision={res['precision']} f1={res['f1']}")
    print(f"reference linework px={res['reference_px']}  candidate px={res['candidate_px']}")
    print(ascii_report(res))
