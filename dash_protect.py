# -*- coding: utf-8 -*-
"""dash_protect.py —— 可落地模块：虚线识别 → 保护掩码。

给主线的接入方式（伪代码）：
    import dash_protect
    res     = dash_protect.find_dashes(sr2)                     # 在擦除前跑一次
    protect = dash_protect.build_protect_mask(sr2.shape, res, mode='ink', confs=('high',))
    clean   = erase_text(sr2, labels, protect=protect)          # wipe &= ~protect

设计取舍：默认 mode='ink'
  只保护"轴线带 ∩ 真实墨像素"。虚线之间的 gap 不保护，带内若混入文字笔画，
  也会被保护（所以模式选择本身是误伤/漏保护的权衡，实测见 REPORT.md）。
"""
from __future__ import annotations
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dash_final as df

# 擦除前的墨层口径。erase_text 内部用的是 (sr2 < 200)，实测在这张图上把 13.3% 的像素
# 判成墨（彩色填充/晕染全进来了），虚线只能检出 high 6 条。
# 改用「暗 + 低色度」——与 cdr_vectorize.measure_text_h 同一套思路，实测与用户的
# ink_before_wipe.png 的 IoU=0.85，且虚线检出 high 12 条。
INK_DARK = 190
INK_CHROMA = 25


def ink_layer(gray, bgr=None, dark=INK_DARK, chroma_max=INK_CHROMA, fallback_thresh=128):
    """擦除前用的"干净墨层"：暗 + 低色度（ink=255）。bgr 为 sr2_color.png。"""
    if bgr is not None:
        lb = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
        chroma = np.hypot(lb[..., 1] - 128, lb[..., 2] - 128)
        return (((gray < dark) & (chroma < chroma_max)).astype(np.uint8)) * 255
    return ((gray < fallback_thresh).astype(np.uint8)) * 255


def find_dashes(gray, bgr=None, **kw):
    """在擦除前的灰度图上识别虚线。返回 dash_final.analyze 的结果 dict。

    gray 必须是 sr2.png（工作尺度灰度，与 erase_text 处理的是同一张）；bgr 是 sr2_color.png。
    传 bw 让线宽用灰度 FWHM 测、识别用干净墨层 —— 两者尺寸一致，不能像调研报告那样
    把 1x 的 src.png 传进来。
    """
    bw = ink_layer(gray, bgr)
    return df.analyze(gray, bw=bw, **kw)


def _segs_of(d, dash, gap):
    """把一条虚线按 dash/gap 展开成若干段（用于 mode='pattern'）。"""
    p0 = np.array(d['p0'], float); p1 = np.array(d['p1'], float)
    L = float(np.linalg.norm(p1 - p0))
    if L < 1:
        return []
    u = (p1 - p0) / L
    out, t = [], 0.0
    step = max(dash + gap, 1.0)
    while t < L:
        out.append((p0 + u * t, p0 + u * min(t + dash, L)))
        t += step
    return out


def build_protect_mask(shape, res, margin=2, confs=('high',), mode='ink', ink=None,
                       kinds=('dashed', 'dashdot')):
    """生成 protect 掩码（bool，True = 禁止擦除）。

    margin : 轴线带在 width_bw 之外再加宽多少 px（工作尺度 px，S=2 时 2px ≈ 源图 1px）
    confs  : 采用哪些置信档，('high',) 最保守
    mode   : 'ink'     = 带 ∩ 真实墨（默认，最安全）
             'band'    = 整条带（含 gap，最激进）
             'pattern' = 按 dash/gap 生成的理想虚线图案 ∩ 墨
    """
    m = np.zeros(shape[:2], np.uint8)
    kept = []
    for d in res['dashes']:
        if d.get('conf') not in confs or d.get('kind') not in kinds:
            continue
        p0 = np.round(np.array(d['p0'])).astype(int)
        p1 = np.round(np.array(d['p1'])).astype(int)
        thick = int(round(d.get('width_bw') or 4)) + 2 * margin
        if mode == 'pattern':
            dash = d.get('dash'); gap = d.get('gap')
            if not np.isfinite(dash) or not np.isfinite(gap) or dash <= 0 or gap <= 0:
                continue
            for a, b in _segs_of(d, dash, gap):
                cv2.line(m, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)),
                         1, max(thick, 1))
        else:
            cv2.line(m, tuple(p0), tuple(p1), 1, max(thick, 1))
        kept.append(d)
    mask = m > 0
    if mode in ('ink', 'pattern') and ink is not None:
        mask &= ink > 0
    return mask, kept


if __name__ == '__main__':
    import json
    lab = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, lab)
    import cdr_vectorize as cv_
    sr2 = cv_.imread(os.path.join(lab, 'work1', 'sr2.png'))
    bgr = cv_.imread(os.path.join(lab, 'work1', 'sr2_color.png'), cv2.IMREAD_COLOR)
    res = find_dashes(sr2, bgr)
    n = {}
    for d in res['dashes']:
        n[d['conf']] = n.get(d['conf'], 0) + 1
    print(f'虚线 {len(res["dashes"])} 条 {n}，实线 {len(res["solids"])} 条，未分类 {len(res["unknown"])} 条')
    for confs in [('high',), ('high', 'medium'), ('high', 'medium', 'low')]:
        for mode in ('ink', 'band', 'pattern'):
            mk, kept = build_protect_mask(sr2.shape, res, confs=confs, mode=mode,
                                          ink=ink_layer(sr2, bgr))
            print(f'  confs={confs} mode={mode:8s} 条数={len(kept):2d} 覆盖 {mk.sum():7d} px '
                  f'({mk.mean()*100:.3f}% 面积)')
