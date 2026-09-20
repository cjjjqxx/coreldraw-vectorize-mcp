# -*- coding: utf-8 -*-
"""
dash_final.py —— 【可直接使用】从含噪、含文字的栅格图中识别 实线 / 虚线 / 点划线。

本文件自包含，只依赖 numpy + opencv-python。所有参数与判据均在本机合成图上实测标定过，
实测结论见同目录 报告.md。

用法：
    import cv2, dash_final
    gray = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    res = dash_final.analyze(gray)
    for d in res['dashes']:
        print(d['p0'], d['p1'], d['dash'], d['gap'], d['width'])   # 直接喂给 CorelDRAW
    dash_final.visualize(gray, res, 'out.png')

设计要点（每条都是实测踩坑后的修正，别改）：
 1. 闭运算**逐方向**做，不要把各方向结果 OR 起来 —— OR 会把交叉的线粘成一坨，
    PCA 主轴拟合随即失效（实测：两条交叉实线直接消失）。
 2. 候选**按轴去重** —— 逐方向会产生同一条线的多个重复片段（实测一条虚线被检出 3 次）。
 3. 轴线用**组内全部像素的 TLS(PCA)** 拟合，不要用"最长段的方向"（实测斜虚线偏 2°）。
 4. 线型用**沿线展开成 1D 剖面 + 游程分析**测，不要用连通域统计
    （HoughLinesP 短段模式会输出大量重叠段，直接算会得到负的间隔）。
 5. 剖面宽度**自适应于线宽**（halfwidth ≈ width/2）—— 固定 ±3px 会把离轴线 10px
    的文字卷进剖面，把一条正常虚线判成 irregular。
 6. dash/gap 需要**腐蚀补偿**（erode_px=2）：自适应二值化把线段两端各膨胀约 1.5px，
    不补偿则 dash 系统性偏大 3px、gap 偏小 3px（实测 4.31px 平均误差 → 1.69px）。
 7. 判 dashdot 要用**未腐蚀**的游程（腐蚀会把点划线的"点"吃掉，实测点划线被误判成虚线）。
"""
from __future__ import annotations
import math
import numpy as np
import cv2

# ---------------------------------------------------------------- 可调参数
P = dict(
    block=31, C=12, open_k=2,        # 自适应二值化
    close_len=30,                    # 方向闭运算核长（≈ 最大 dash+gap 周期 + 余量）
    close_thick=3,                   # 闭运算核粗细
    min_axis=60,                     # 认作一条线的短边下限（px）
    min_aspect=5.0,                  # 长宽比下限
    ang_tol=12.0,                    # 候选方向与核方向的容差（度）
    erode_px=2,                      # dash/gap 的膨胀补偿量
    hough_thr=40, hough_minlen=80,   # 实线（Hough 长段）
    hough_maxgap=3, hough_angtol=3.0, hough_pertol=4.0, hough_minlen_out=80,
)


# ---------------------------------------------------------------- 二值化
def binarize(gray, block=None, C=None, open_k=None):
    p = P
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV,
                               block or p['block'], C if C is not None else p['C'])
    k = open_k if open_k is not None else p['open_k']
    if k:
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    return bw


# ---------------------------------------------------------------- 连通域特征
def comp_features(bw, min_area=30):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(bw, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        sub = (lab[y:y + h, x:x + w] == i).astype(np.uint8)
        ys, xs = np.nonzero(sub)
        if xs.size < 5:
            continue
        pts = np.column_stack([xs + x, ys + y]).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        d = np.array([vx, vy], float); d /= (np.linalg.norm(d) + 1e-9)
        t = (pts - np.array([x0, y0], float)) @ d
        length = float(t.max() - t.min())
        rw, rh = cv2.minAreaRect(pts)[1]
        aspect = max(rw, rh) / max(min(rw, rh), 1e-6)
        out.append(dict(id=i, area=int(area), bbox=(int(x), int(y), int(w), int(h)),
                        cx=float(cent[i][0]), cy=float(cent[i][1]),
                        p0=(float(x0 + d[0] * t.min()), float(y0 + d[1] * t.min())),
                        p1=(float(x0 + d[0] * t.max()), float(y0 + d[1] * t.max())),
                        dir=d, theta=float(math.atan2(d[1], d[0]) % math.pi),
                        length=length, aspect=float(aspect)))
    return out


def _ang_diff(a, b):
    d = abs((a % math.pi) - (b % math.pi))
    return min(d, math.pi - d)


def line_kernel(L, angle_deg, thick=3):
    L = max(3, int(L))
    k = np.zeros((L, L), np.uint8)
    c = L // 2
    r = math.radians(angle_deg)
    cv2.line(k, (int(round(c - math.cos(r) * c)), int(round(c - math.sin(r) * c))),
             (int(round(c + math.cos(r) * c)), int(round(c + math.sin(r) * c))), 1, thick)
    return k


# ---------------------------------------------------------------- 候选（逐方向闭运算）
def candidates(bw, angles=None, **kw):
    p = dict(P); p.update(kw)
    angles = angles if angles is not None else range(0, 180, 5)
    cands = []
    for a in angles:
        k = line_kernel(p['close_len'], a, p['close_thick'])
        if k.sum() == 0:
            continue
        closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        for f in comp_features(closed, min_area=30):
            if f['length'] < p['min_axis'] or f['aspect'] < p['min_aspect']:
                continue
            if math.degrees(_ang_diff(f['theta'], math.radians(a))) > p['ang_tol']:
                continue
            cands.append(f)
    return dedup_by_axis(cands)


def dedup_by_axis(cands, ang_tol=5.0, perp_tol=3.0, overlap=0.5):
    """同轴去重：角度接近 + 垂距接近 + 沿轴重叠 → 只留最长的那个。"""
    keep = []
    for f in sorted(cands, key=lambda z: -z['length']):
        dup = False
        for g in keep:
            if math.degrees(_ang_diff(f['theta'], g['theta'])) > ang_tol:
                continue
            u = g['dir']; n = np.array([-u[1], u[0]])
            c1 = np.array([f['cx'], f['cy']]); c2 = np.array([g['cx'], g['cy']])
            if abs(float((c1 - c2) @ n)) > perp_tol:
                continue
            a0 = np.array(f['p0']) @ u; a1 = np.array(f['p1']) @ u
            b0 = np.array(g['p0']) @ u; b1 = np.array(g['p1']) @ u
            ov = min(max(a0, a1), max(b0, b1)) - max(min(a0, a1), min(b0, b1))
            if ov > overlap * min(f['length'], g['length']):
                dup = True; break
        if not dup:
            keep.append(f)
    return keep


# ---------------------------------------------------------------- 轴线精拟合
def refine_axis(bw, f, band=3):
    """用该候选带内的**原始 ink 像素**做 TLS(PCA) 拟合 → 精确轴线。"""
    m = np.zeros_like(bw)
    cv2.polylines(m, [np.round(np.array([f['p0'], f['p1']])).astype(np.int32)], False, 255, band)
    pts = np.column_stack(np.nonzero(bw & m)[::-1]).astype(np.float32)
    if pts.shape[0] < 10:
        return None
    c = pts.mean(0)
    _u, _s, vt = np.linalg.svd(pts - c, full_matrices=False)
    d = vt[0] / (np.linalg.norm(vt[0]) + 1e-9)
    t = (pts - c) @ d
    p0 = c + d * float(t.min()); p1 = c + d * float(t.max())
    return (float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))


# ---------------------------------------------------------------- 1D 剖面
def unwrap(bw, p0, p1, halfwidth=3):
    p0 = np.array(p0, float); p1 = np.array(p1, float)
    v = p1 - p0
    L = int(np.linalg.norm(v)) + 1
    if L < 2:
        return np.zeros((2, 2 * halfwidth + 1), np.uint8)
    d = v / np.linalg.norm(v)
    n = np.array([-d[1], d[0]])
    s = np.arange(L, dtype=np.float32)
    o = np.arange(-halfwidth, halfwidth + 1, dtype=np.float32)
    X = (p0[0] + d[0] * s[:, None] + n[0] * o[None, :]).astype(np.float32)
    Y = (p0[1] + d[1] * s[:, None] + n[1] * o[None, :]).astype(np.float32)
    return cv2.remap(bw, X, Y, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def run_lengths(profile):
    p = profile.astype(np.int8)
    d = np.diff(np.concatenate([[0], p, [0]]))
    st = np.flatnonzero(d == 1); en = np.flatnonzero(d == -1)
    segs = list(zip(st.tolist(), en.tolist()))
    on = np.array([e - s for s, e in segs], float)
    off = np.array([segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)], float) \
        if len(segs) > 1 else np.array([])
    return on, off, segs


def thickness_bw(bw, p0, p1, search=8):
    """二值图上的横向厚度中位数（会被二值化膨胀，偏大）。向量化实现。"""
    strip = unwrap(bw, p0, p1, search)
    if strip.shape[0] < 5:
        return 0.0
    cnt = (strip > 0).sum(axis=1).astype(float)
    on = cnt[cnt > 0]
    return float(np.median(on)) if on.size else 0.0


def thickness_fwhm(gray, p0, p1, search=8):
    """灰度剖面半高全宽 —— 更接近真实线宽（实测真值 2px 时给 3.0~3.25px，二值化给 4~5.5px）。向量化实现。"""
    if gray is None:
        return 0.0
    strip = unwrap(gray, p0, p1, search).astype(np.float32)
    if strip.shape[0] < 5:
        return 0.0
    bg = np.percentile(strip, 85, axis=1)
    mn = strip.min(axis=1)
    ok = (bg - mn) > 40
    if ok.sum() < 5:
        return 0.0
    half = (bg + mn) / 2.0
    cnt = (strip < half[:, None]).sum(axis=1).astype(float)
    return float(np.median(cnt[ok]))


def measure(bw, p0, p1, halfwidth, erode_px=2, min_runs=3, min_dash=3, band=1):
    strip = unwrap(bw, p0, p1, halfwidth)
    hw = strip.shape[1] // 2
    band = min(band, hw)
    prof = (strip[:, hw - band:hw + band + 1].max(axis=1) > 0).astype(np.uint8)
    prof_raw = prof.copy()
    if erode_px > 0:
        prof = cv2.erode(prof.reshape(1, -1), np.ones((1, 2 * erode_px + 1), np.uint8)).ravel()
    on, off, segs = run_lengths(prof)
    res = dict(kind='none', dash=float('nan'), gap=float('nan'), duty=float(prof.mean()),
               n_seg=len(segs), gap_cv=float('nan'), len_cv=float('nan'))
    if len(segs) == 0:
        return res, prof, prof_raw
    if res['duty'] > 0.90:
        res['kind'] = 'solid'
        return res, prof, prof_raw
    core_on = on[1:-1] if len(on) > 2 else on
    core_off = off[1:-1] if len(off) > 2 else off
    res['dash'] = float(np.median(core_on))
    res['gap'] = float(np.median(core_off)) if core_off.size else 0.0
    res['len_cv'] = float(core_on.std() / (core_on.mean() + 1e-6)) if core_on.size else 9.9
    res['gap_cv'] = float(core_off.std() / (core_off.mean() + 1e-6)) if core_off.size else 9.9
    if len(segs) < min_runs or res['dash'] < min_dash:
        res['kind'] = 'sparse'
        return res, prof, prof_raw
    # 点划线：用**未腐蚀**的游程判（腐蚀会把"点"吃掉）
    on_raw, off_raw, _ = run_lengths(prof_raw)
    core_raw = on_raw[1:-1] if len(on_raw) > 2 else on_raw
    core_off_raw = off_raw[1:-1] if len(off_raw) > 2 else off_raw
    med_raw = float(np.median(core_raw)) if core_raw.size else 0
    if core_raw.size >= 4 and med_raw > 0 and \
            core_raw.max() > 2.5 * med_raw and core_raw.min() < 0.5 * med_raw:
        res['kind'] = 'dashdot'
        # 长划 / 点 / 间隔 —— CDR 的 dash-dot 样式需要这三个量
        res['long'] = float(np.percentile(core_raw, 75))
        res['dot'] = float(np.percentile(core_raw, 25))
        res['gap'] = float(np.median(core_off_raw)) if core_off_raw.size else 0.0
        res['dash'] = res['long']
    elif res['len_cv'] < 0.40 and res['gap_cv'] < 0.40:
        # ---- 合理性约束（真实地质图实测：不加会出严重假阳性）----
        # 实测反例：4 个分散段被连成 "d=17.5 g=172 n=4"，gap 是段长的 10 倍，
        #           占空比只有 0.12 —— 那根本不是虚线。
        ratio = res['gap'] / max(res['dash'], 1.0)
        if len(segs) < 4 or res['duty'] < 0.20 or ratio > 3.0:
            res['kind'] = 'irregular'
            res['why'] = (f"n={len(segs)} duty={res['duty']:.2f} "
                          f"gap/dash={ratio:.1f}（需 n>=4, duty>=0.20, gap/dash<=3）")
        else:
            res['kind'] = 'dashed'
    else:
        res['kind'] = 'irregular'
    return res, prof, prof_raw


# ---------------------------------------------------------------- 实线（Hough 长段 + 轴去重）
def hough_solids(bw, **kw):
    p = dict(P); p.update(kw)
    ls = cv2.HoughLinesP(bw, 1, np.pi / 720, p['hough_thr'],
                         minLineLength=p['hough_minlen'], maxLineGap=p['hough_maxgap'])
    if ls is None:
        return []
    # cv2 >= 5.0 returns (N,4); cv2 4.x returned (N,1,4) -- ravel() handles both
    segs = [tuple(map(float, np.asarray(l).ravel()[:4])) for l in ls]
    used = [False] * len(segs); groups = []
    for i, s in enumerate(segs):
        if used[i]:
            continue
        a = math.atan2(s[3] - s[1], s[2] - s[0]) % math.pi
        u = np.array([math.cos(a), math.sin(a)]); n = np.array([-u[1], u[0]])
        c1 = np.array([(s[0] + s[2]) / 2, (s[1] + s[3]) / 2])
        g = [i]; used[i] = True
        for j in range(i + 1, len(segs)):
            if used[j]:
                continue
            b = math.atan2(segs[j][3] - segs[j][1], segs[j][2] - segs[j][0]) % math.pi
            if math.degrees(_ang_diff(a, b)) > p['hough_angtol']:
                continue
            c2 = np.array([(segs[j][0] + segs[j][2]) / 2, (segs[j][1] + segs[j][3]) / 2])
            if abs(float((c2 - c1) @ n)) > p['hough_pertol']:
                continue
            g.append(j); used[j] = True
        groups.append((a, g, u, n))
    out = []
    for a, g, u, n in groups:
        iv = []
        for i in g:
            x1, y1, x2, y2 = segs[i]
            t1 = np.array([x1, y1]) @ u; t2 = np.array([x2, y2]) @ u
            iv.append((min(t1, t2), max(t1, t2)))
        iv.sort()
        merged = [list(iv[0])]
        for lo, hi in iv[1:]:
            if lo <= merged[-1][1] + 5:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        perp = float(np.median([np.array([(segs[i][0] + segs[i][2]) / 2,
                                          (segs[i][1] + segs[i][3]) / 2]) @ n for i in g]))
        for lo, hi in merged:
            if hi - lo < p['hough_minlen_out']:
                continue
            q0 = u * lo + n * perp; q1 = u * hi + n * perp
            out.append(dict(p0=(float(q0[0]), float(q0[1])), p1=(float(q1[0]), float(q1[1])),
                            length=float(hi - lo), theta=float(a)))
    return out


# ---------------------------------------------------------------- 主入口
def analyze(gray, bw=None, use_hough_solids=True, **kw):
    """
    返回 dict:
      dashes  : 虚线/点划线 [{p0,p1,dash,gap,width,width_bw,duty,n_seg,kind,conf}]
      solids  : 实线     [{p0,p1,width,width_bw,length}]
      unknown : 未能分类的长条（可能是曲线/文字行/点填充区）
      profile : 每条虚线的 1D 剖面（调试用）
    p0/p1 是**原图像素坐标**，dash/gap/width 单位是像素（在 CorelDRAW 里按比例换算）。
    """
    p = dict(P); p.update(kw)
    if bw is None:
        bw = binarize(gray, p['block'], p['C'], p['open_k'])
    dashes, solids, unknown, profiles = [], [], [], []

    for f in candidates(bw, None, **p):
        ax = refine_axis(bw, f, band=3)
        if ax is None:
            continue
        p0, p1 = ax
        w_bw = thickness_bw(bw, p0, p1)
        w_fwhm = thickness_fwhm(gray, p0, p1)
        hw = int(max(1, round(w_bw / 2.0)))          # 关键：自适应剖面半宽
        pat, prof, prof_raw = measure(bw, p0, p1, hw, erode_px=p['erode_px'])
        rec = dict(p0=p0, p1=p1, dash=pat['dash'], gap=pat['gap'], duty=pat['duty'],
                   n_seg=pat['n_seg'], gap_cv=pat['gap_cv'], len_cv=pat['len_cv'],
                   long=pat.get('long', float('nan')), dot=pat.get('dot', float('nan')),
                   width=round(w_fwhm if w_fwhm > 0 else w_bw, 1),
                   width_bw=round(w_bw, 1), kind=pat['kind'],
                   length=float(math.hypot(p1[0] - p0[0], p1[1] - p0[1])))
        if pat['kind'] == 'solid' or pat['duty'] > 0.90:
            rec['kind'] = 'solid'; solids.append(rec)
        elif pat['kind'] in ('dashed', 'dashdot'):
            # 置信度：三条判据同时满足才 high（真实图上实测区分度最好的组合）
            ratio = rec['gap'] / max(rec['dash'], 1.0)
            if pat['gap_cv'] < 0.25 and pat['len_cv'] < 0.30 and rec['duty'] >= 0.35 and ratio <= 2.0:
                rec['conf'] = 'high'
            elif pat['gap_cv'] < 0.40 and pat['len_cv'] < 0.40 and rec['duty'] >= 0.25 and ratio <= 3.0:
                rec['conf'] = 'medium'
            else:
                rec['conf'] = 'low'
            dashes.append(rec); profiles.append(prof)
        else:
            if 'why' in pat:
                rec['why'] = pat['why']
            unknown.append(rec)

    if use_hough_solids:
        # Hough 补漏：闭运算对"交叉的实线"无能力（实测两条交叉实线会消失）
        for s in hough_solids(bw, **p):
            dup = False
            for g in solids:
                if math.degrees(_ang_diff(s['theta'], math.atan2(g['p1'][1] - g['p0'][1],
                                                                 g['p1'][0] - g['p0'][0]))) > 5:
                    continue
                u = np.array([math.cos(s['theta']), math.sin(s['theta'])])
                n = np.array([-u[1], u[0]])
                c1 = np.array([(s['p0'][0] + s['p1'][0]) / 2, (s['p0'][1] + s['p1'][1]) / 2])
                c2 = np.array([(g['p0'][0] + g['p1'][0]) / 2, (g['p0'][1] + g['p1'][1]) / 2])
                if abs(float((c1 - c2) @ n)) < 4:
                    dup = True; break
            if not dup:
                w = thickness_fwhm(gray, s['p0'], s['p1'])
                solids.append(dict(p0=s['p0'], p1=s['p1'], dash=float('nan'), gap=float('nan'),
                                   duty=1.0, n_seg=1, width=round(w, 1),
                                   width_bw=round(thickness_bw(bw, s['p0'], s['p1']), 1),
                                   kind='solid', length=s['length'], source='hough'))
    return dict(dashes=dashes, solids=solids, unknown=unknown, profiles=profiles, bw=bw)


# ---------------------------------------------------------------- 可视化
def visualize(gray, res, path=None, scale=1.0):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for s in res['solids']:
        cv2.line(vis, tuple(np.round(s['p0']).astype(int)), tuple(np.round(s['p1']).astype(int)),
                 (0, 170, 0), 2)
    for d in res['dashes']:
        cv2.line(vis, tuple(np.round(d['p0']).astype(int)), tuple(np.round(d['p1']).astype(int)),
                 (0, 0, 255), 2)
        cv2.putText(vis, f"{d['kind']} d={d['dash']:.0f} g={d['gap']:.0f} w={d['width']:.1f}",
                    (int(d['p0'][0]), int(d['p0'][1]) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 220), 1, cv2.LINE_AA)
    for u in res['unknown']:
        cv2.line(vis, tuple(np.round(u['p0']).astype(int)), tuple(np.round(u['p1']).astype(int)),
                 (200, 200, 200), 1)
    if scale != 1.0:
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    if path:
        cv2.imwrite(path, vis)
    return vis


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        try:
            import synth
            gray, _bgr, _t = synth.build()
        except Exception:
            print("用法: python dash_final.py <图片路径>"); raise SystemExit(1)
    else:
        gray = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    res = analyze(gray)
    print(f"虚线/点划线 {len(res['dashes'])} 条，实线 {len(res['solids'])} 条，未分类 {len(res['unknown'])} 条")
    for d in res['dashes']:
        print(f"  {d['kind']:8s} ({d['p0'][0]:.0f},{d['p0'][1]:.0f})→({d['p1'][0]:.0f},{d['p1'][1]:.0f}) "
              f"dash={d['dash']:.1f} gap={d['gap']:.1f} width={d['width']:.1f} n={d['n_seg']} conf={d.get('conf')}")
    for s in res['solids']:
        print(f"  solid    ({s['p0'][0]:.0f},{s['p0'][1]:.0f})→({s['p1'][0]:.0f},{s['p1'][1]:.0f}) "
              f"width={s['width']:.1f} len={s['length']:.0f} src={s.get('source','cc')}")
    visualize(gray, res, 'out/dash_final_vis.png')
    print("可视化: out/dash_final_vis.png")
