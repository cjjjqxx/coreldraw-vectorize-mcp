"""Rasterise the vector model (layers.json) in Python, without CorelDRAW.

Every check so far had to wait for the CorelDRAW step - 50 to 150 seconds - before anything could be
measured, so tuning happened by eye on one image at a time. Rendering the same model here takes about a
second, which makes two things possible:

  * the pipeline can check ITSELF before it draws: if the model is already wrong against the source,
    drawing it in CorelDRAW only spends two minutes to produce the same wrong picture;
  * parameters can be fitted against a measured objective instead of hand-tuned.

It draws what the COM side draws: region shapes (filled Bezier paths with holes, optional outline),
stroke layers (centre lines, dashed where the model says so) and, for the layer pipeline, the layer
masks in paint order. Text is not drawn - the structural check excludes the label boxes anyway.
"""
from __future__ import annotations

import math
import os

import cv2
import numpy as np


def _bezier(p0, seg, steps=12):
    """Cubic segment (x, y, c1x, c1y, c2x, c2y) -> points, p0 included."""
    x, y, c1x, c1y, c2x, c2y = seg
    t = np.linspace(0.0, 1.0, steps)[:, None]
    p0 = np.array(p0, np.float64)
    c1 = np.array([c1x, c1y], np.float64)
    c2 = np.array([c2x, c2y], np.float64)
    p1 = np.array([x, y], np.float64)
    pts = ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * c1 + 3 * (1 - t) * (t ** 2) * c2 + (t ** 3) * p1
    return pts


def _path_points(path, scale=1.0):
    pts = [np.array(path['start'], np.float64)]
    cur = pts[0]
    for seg in path['segs']:
        p = _bezier(cur, seg)
        pts.append(p[1:])
        cur = p[-1]
    out = np.vstack(pts) * scale
    return np.round(out).astype(np.int32)


def render(meta, size, scale=1.0, bg=(255, 255, 255)):
    """meta = layers.json, size = (w, h) in the meta's own working px. Returns a BGR image."""
    w, h = size
    img = np.full((int(h * scale), int(w * scale), 3), bg, np.uint8)
    for lay in meta.get('layers', []):
        kind = lay.get('kind')
        if kind == 'regions':
            # fills first, outlines after: a frame drawn as a ring region is painted early (it is the
            # largest shape) and every later fill covered its dark edge - that is how the figure's own
            # border went missing while every metric still looked fine
            later = []
            for sh in lay.get('shapes', []):
                paths = sh.get('paths') or [{'start': sh.get('start'), 'segs': sh.get('segs') or []}]
                polys = [_path_points(p, scale) for p in paths if p.get('start') and p.get('segs')]
                if not polys:
                    continue
                cv2.fillPoly(img, polys, tuple(int(v) for v in sh['color'][::-1]), lineType=cv2.LINE_AA)
                if sh.get('outline'):
                    later.append((polys, sh['outline']))
            for polys, ol in later:
                cv2.polylines(img, polys, True, tuple(int(v) for v in ol['color'][::-1]),
                              max(1, int(round(float(ol['width_px']) * scale))), lineType=cv2.LINE_AA)
            continue
        st = lay.get('strokes')
        if st:
            colour = tuple(int(v) for v in st['color'][::-1])
            width = max(1, int(round(float(st['width_px']) * scale)))
            dash = st.get('dash')
            for pl in st['polylines']:
                pts = np.round(np.array(pl, np.float64).reshape(-1, 2) * scale).astype(np.int32)
                if len(pts) < 2:
                    continue
                if dash:
                    _dashed(img, pts, colour, width, dash[0] * scale, dash[1] * scale)
                else:
                    cv2.polylines(img, [pts], False, colour, width, lineType=cv2.LINE_AA)
            continue
    rules = meta.get('rules')
    if rules and rules.get('polylines'):
        colour = tuple(int(v) for v in rules['color'][::-1])
        width = max(1, int(round(float(rules['width_px']) * scale)))
        for pl in rules['polylines']:
            pts = np.round(np.array(pl, np.float64).reshape(-1, 2) * scale).astype(np.int32)
            if len(pts) >= 2:
                cv2.polylines(img, [pts], False, colour, width, lineType=cv2.LINE_AA)
    return img


def _dashed(img, pts, colour, width, dash, gap):
    """Draw a polyline with a dash pattern (CorelDRAW draws the same line with an outline style)."""
    if dash <= 0:
        cv2.polylines(img, [pts], False, colour, width, lineType=cv2.LINE_AA)
        return
    period = max(dash + gap, 1.0)
    for i in range(len(pts) - 1):
        a, b = pts[i].astype(np.float64), pts[i + 1].astype(np.float64)
        L = float(np.linalg.norm(b - a))
        if L < 1e-6:
            continue
        u = (b - a) / L
        t = 0.0
        while t < L:
            t1 = min(t + dash, L)
            p0 = (a + u * t).astype(np.int32)
            p1 = (a + u * t1).astype(np.int32)
            cv2.line(img, tuple(p0), tuple(p1), colour, width, lineType=cv2.LINE_AA)
            t += period


def render_work_dir(work_dir, out_name='model_render.png', ss=3):
    """Render the model of a work dir at source resolution; returns the image (and writes it).

    Supersampled: drawn at `ss` times the size and averaged down. A sub-pixel line drawn directly at 1:1
    has to be rounded up to a whole pixel, which made every hairline solid and the renderer far more
    generous than CorelDRAW (mark recall 0.968 here against 0.896 for the same model actually drawn).
    """
    import json
    meta = json.load(open(os.path.join(work_dir, 'layers.json'), encoding='utf-8'))
    w, h = meta['src_size']
    S = meta.get('scale') or 1
    big = render(meta, (w * S, h * S), scale=float(ss) / S)
    img = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
    if out_name:
        cv2.imencode('.png', img)[1].tofile(os.path.join(work_dir, out_name))
    return img
