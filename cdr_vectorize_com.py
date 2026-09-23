"""CorelDRAW half of the vectorize pipeline: PowerTRACE sr2_no_text.png, put labels back as
real text, save .cdr + .png. Run as a subprocess (COM can block forever on a modal dialog).

usage: python cdr_vectorize_com.py <work_dir> [font] [trace_type] [detail] [smoothing]
prints one JSON line with the result.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback

import pythoncom

# '宋体' is broken through COM in CorelDRAW 2022 here: it becomes 'yyb' at creation and renders
# tofu when set on the story. These were verified to render CJK correctly.
FONT_PREF = ['新宋体', '仿宋', '黑体', '微软雅黑']
TILE_TRACE = os.environ.get('CDR_NO_TILE_TRACE', '').strip() in ('', '0', 'false', 'False')
TRACE_TYPES = {'lineart': 1, 'logo': 2, 'detailed_logo': 3, 'technical': 7, 'line_drawing': 8}

# Font sizing from the label's own measured text run, computed by cdr_vectorize_color. On by default: it
# touches only the font size, and the alternative - sizing a rotated label from its axis-aligned box -
# makes the font follow the box diagonal, which is the giant-label defect. CDR_NO_MEASURED_TEXT=1 turns
# it off. Read from the environment so this child process need not import the whole vectorize module.
MEASURED_TEXT = not (os.environ.get('CDR_NO_MEASURED_TEXT', '').strip().lower() in ('1', 'true', 'yes')) \
    or (os.environ.get('CDR_EXPERIMENTAL', '').strip().lower() in ('1', 'true', 'yes'))



def group_children(grp):
    """Shapes of a trace result. A layer that traces to ONE shape comes back as a plain curve whose
    Shapes.Count is 0 (no exception): treating that as "no children" left it uncoloured, i.e. black."""
    try:
        n = int(grp.Shapes.Count)
    except Exception:  # noqa: BLE001 - not a group
        return [grp]
    return [grp.Shapes.Item(i) for i in range(1, n + 1)] if n > 0 else [grp]


def drop_degenerate(shapes, min_span=0.0):
    """PowerTRACE sometimes emits zero-area strokes (a 2-node line from the page corner to mid-drawing).
    Invisible as fills, they show as stray diagonal lines once outlined or traced at higher resolution.
    Area only: a small oval patch is a valid closed curve of just 2 nodes."""
    keep = []
    for sh in shapes:
        try:
            # zero area alone is not enough: CorelDRAW also reports 0 for tiny dots and holed shapes,
            # deleting those lost real symbols. The spikes are long (hundreds of px).
            if float(sh.Curve.Area) <= 0 and max(sh.SizeWidth, sh.SizeHeight) > min_span:
                sh.Delete()
                continue
        except Exception:  # noqa: BLE001 - not a curve / no area
            pass
        keep.append(sh)
    return keep


def _trace_mask(app, doc, layer, png_path, page_w, page_h, trace_type, detail, smoothing, rgb=None,
                outline=None, k=None):
    """Import a black-on-white mask, PowerTRACE it, optionally recolour the result. Returns the group."""
    layer.Import(png_path, 0, app.CreateStructImportOptions())
    bmp = doc.ActiveShape
    bmp.SetSize(page_w, page_h); bmp.LeftX = 0; bmp.BottomY = 0
    ts = bmp.Bitmap.Trace(TRACE_TYPES.get(trace_type, 1), smoothing, detail, 8, 0, 2, True, True, True)
    ts.DetailLevelPercent = detail; ts.Smoothing = smoothing; ts.CornerSmoothness = 0
    ts.MergeAdjacentObjects = True
    ts.ApplyChanges()
    stats = (int(ts.CurveCount), int(ts.NodeCount))
    ts.Finish()
    grp = doc.ActiveShape
    if rgb is not None:
        col = app.CreateColor(); col.RGBAssign(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        shapes = group_children(grp)
        shapes = drop_degenerate(shapes, min_span=20 * k if k else 0.0) if outline else shapes
        for sh in shapes:
            try:
                sh.Fill.ApplyUniformFill(col)
                if outline and k:
                    # colour first: traced shapes have no outline, and assigning a colour creates one with
                    # the 0.2 in default width - a width set before that was silently dropped
                    sh.Outline.Color.RGBAssign(*[int(v) for v in outline['color']])
                    sh.Outline.Width = float(outline['width_px']) * k
                    sh.Outline.LineJoin = 1                                     # round: no miter spikes
                else:
                    sh.Outline.SetNoOutline()
            except Exception:  # noqa: BLE001
                pass
    return grp, stats


def label_lang(lab):
    """Language ID for a text object. 2052 (Chinese) applies Asian spacing even to Latin fonts: "0.6"
    came out 17% wider than the same text as 1033 and ran into axis ticks."""
    import re
    return 2052 if re.search(r'[　-鿿＀-￯]', lab['text']) else 1033


def fit_measured_text(sh, lab, k):
    """Size a SLANTED text object from the run measured along its own line, not from the label box.

    The box is an axis-aligned container around rotated text, and recovering the text length from it is
    a singular inversion at 45 degrees (the fallback max(W,H) made the font follow the box: giants).
    The run length L is measured by projecting the glyph ink onto the label's own direction, so the
    box plays no part.

    Width only. This used to go on and rescale the object so its height matched the measured g - but g
    is the span holding 80% of the ink across the line, which for lowercase words is about the x-height,
    while SizeHeight is the whole line with ascenders and descenders. Comparing the two halved every
    clean label (slope, Tuozitong, Guanghan came out at half size) and, where a crossing line inflated
    g, blew labels up instead (the giants). It was also applied to horizontal labels, which the tuned
    fit_horizontal_text handled before: the regression's median size error went from 0.11 to 0.24.
    Returns False when nothing usable was measured, so the caller keeps the box-derived path.
    """
    m = lab.get('measured') or {}
    try:
        length = float(m.get('L') or 0.0)
    except Exception:  # noqa: BLE001
        return False
    if length < 4.0:
        return False
    try:
        if sh.SizeWidth <= 0:
            return False
        # L is the narrowest span holding 80% of the run's ink (robust to strays), so for evenly spread
        # lettering it is 0.8 of the run: every slanted label came out at ~80% size. Measured on the
        # clean labels of the geology map, full run / L = 1.26-1.31.
        sh.Text.Story.Size = float(sh.Text.Story.Size) * (1.25 * length * k) / float(sh.SizeWidth)
        # Same tracked-out lettering guard as fit_rotated_text, but generous: g is the measured
        # x-height and SizeHeight the whole line (ascender to descender), a ratio of ~2.4 for normal
        # type, so only a gross overshoot is pulled back.
        cap_h = size_cap(lab, k, 2.0 * float(m.get('g') or 0.0) * k)
        if cap_h > 0 and float(sh.SizeHeight) > cap_h:
            sh.Text.Story.Size = float(sh.Text.Story.Size) * cap_h / float(sh.SizeHeight)
    except Exception:  # noqa: BLE001 - a cosmetic fit must never fail a build
        return False
    return True


def create_label(layer, lab, cx, cy, font, meta, fonts, k, S):
    """Create the text object for a horizontal label. For Latin-only labels the letter spacing is measured
    rather than assumed: Asian spacing (language 2052) is 17% wider than Latin spacing (1033). Tick numbers
    like "0.6" need 1033 (2052 ran them into the axis tick); the maps' bold degree labels need 2052 (1033
    made them taller after the width fit, into the map frame). Both are created and fitted to the located
    width; the one whose height matches the measured source glyph height is kept."""
    fnt = label_font(lab, font, meta, fonts)
    # measured on the regression + holdout set: choosing per label by height was not reliable (small map
    # -4 points text accuracy) and Latin spacing alone cost the maps 4-10 points; Asian spacing everywhere
    # is the best overall. Tick numbers that still collide are reported by the self-check.
    langs = [2052]
    best, best_err = None, None
    for lang in langs:
        sh = layer.CreateArtisticTextWide(cx, cy, lab['text'], lang, 1, 'Arial', 10, False, False, 0, 1)
        sh.Text.Story.Font = fnt
        x0, _, x1, _ = lab['tight']
        if (x1 - x0) > 0 and sh.SizeWidth > 0:
            sh.Text.Story.Size = sh.Text.Story.Size * (x1 - x0) * k / sh.SizeWidth
        th = float(lab.get('text_h') or 0) * S * k
        err = abs(sh.SizeHeight / th - 1.0) if th > 0 and sh.SizeHeight > 0 else 0.0
        if best is None or err < best_err:
            if best is not None:
                best.Delete()
            best, best_err = sh, err
        else:
            sh.Delete()
    return best


def label_font(lab, font, meta, fonts):
    """CJK font for labels with Chinese characters, the detected proportional Latin font otherwise."""
    import re
    # per-label first: a map that prints its region names sans and its place names serif needs both
    latin = lab.get('latin_font') or meta.get('latin_font')
    if latin and latin in fonts and not re.search(r'[　-鿿＀-￯]', lab['text']):
        return latin
    return font


def fit_horizontal_text(sh, lab, k, S):
    """Size a horizontal label: fit the located text width, then check it against the glyph height
    measured in the source. The located width is wrong when the box collapsed onto a fragment (text
    came out 1/4 size) or swallowed a frame line (text twice too big); height wins then."""
    x0, _, x1, _ = lab['tight']
    bw = (x1 - x0) * k
    if bw > 0 and sh.SizeWidth > 0:
        sh.Text.Story.Size = sh.Text.Story.Size * bw / sh.SizeWidth
    th = float(lab.get('text_h') or 0) * S * k            # source px -> page units
    if th > 0 and sh.SizeHeight > 0:
        r = sh.SizeHeight / th
        # CJK: only gross disagreement - the measured height is noisy (radicals, underlines, symbols in the
        # box); 0.7-1.35 overrode good width fits and enlarged underlined labels into their neighbours.
        # Latin/digits: the height is reliable, and the width is the unreliable part (a tick mark or axis
        # stub attached to "0.5" made it 1.5x too big), so a tight band.
        # (a tighter band for Latin labels was tried: fixed axis-tick numbers in one chart but cost 8-10
        # points of text accuracy on the maps, whose bold degree labels it shrank - reverted)
        if r > 1.8 or r < 0.5:
            sh.Text.Story.Size = sh.Text.Story.Size * th / sh.SizeHeight
            # a contaminated height (neighbouring row, symbol) must never push the text far past its
            # own box: that overlapped other labels and ran off the page
            boxes = [lab['tight']] + ([lab['box']] if 'box' in lab else [])
            max_w = 1.15 * max(b[2] - b[0] for b in boxes) * k
            if sh.SizeWidth > max_w > 0:
                sh.Text.Story.Size = sh.Text.Story.Size * max_w / sh.SizeWidth
            return 'height'
    return 'width'


def size_cap(lab, k, g_fallback=0.0):
    import re
    """Upper bound for a text object's SizeHeight, in page units.

    Two independent measures of how big the printed lettering is, both taken from the source:
      * the detector polygon's thickness - the whole printed line, which is what SizeHeight measures;
      * run_gh, the median blob size across the text line, which a line running along the label cannot
        move (the span-based g can: Kang-Dian measured 34 for 11 px letters).
    Tracked-out lettering ("K a n g - D i a n" spread over 200 px) is why a cap is needed at all:
    sizing by the length of the run then makes every glyph about three times too big.
    """
    caps = []
    qt = quad_thickness(lab, k)
    if qt:
        caps.append(1.15 * qt)
    gh = float(lab.get('run_gh') or 0.0) * k
    # CJK glyphs split into radicals, so the median blob is a fraction of the character and this cap
    # would squash them (柴达木盆地 measured 8.8 px for 38 px characters). Latin only.
    if gh > 0 and not re.search(r'[　-鿿＀-￯]', str(lab.get('text') or '')):
        caps.append(2.0 * gh)            # SizeHeight ~ 1.15 em, run_gh ~ 0.6 em
    if not caps and g_fallback > 0:
        caps.append(1.3 * g_fallback)
    return min(caps) if caps else 0.0


def quad_thickness(lab, k):
    """Thickness of the detector polygon in page units, or None.

    Map labels are often printed with the letters tracked out ("Kang-Dian" spans 200 px for 9 small
    letters). Sizing such a label by the LENGTH of its run then makes the glyphs about three times too
    big. The polygon hugs the lettering across the line, so it bounds the glyph size where the label
    box - which may also contain the line running beside the text - does not.
    """
    import math
    q = lab.get('quad')
    if not q or len(q) != 4:
        return None
    e = [math.hypot(q[i][0] - q[(i + 1) % 4][0], q[i][1] - q[(i + 1) % 4][1]) for i in range(4)]
    e.sort()
    t = 0.5 * (e[0] + e[1]) * k
    return t if t > 0 else None


def fit_rotated_text(sh, lab, k):
    """Size a slanted / vertical label before rotating it. The located box is axis-aligned around the
    rotated text: for |angle| near 90 the text length is the box height; for a moderate slant solve
    W = L cos + g sin, H = L sin + g cos for the text length L and glyph height g."""
    import math
    ang = float(lab.get('angle') or 0.0)
    x0, y0, x1, y1 = lab['tight']
    W, H = (x1 - x0) * k, (y1 - y0) * k
    t = math.radians(min(abs(ang), 90.0))
    if abs(ang) >= 75:
        L, g = H, W
    else:
        c, s_ = math.cos(t), math.sin(t)
        den = c * c - s_ * s_
        L = (W * c - H * s_) / den if den > 0.2 else max(W, H)
        g = (H * c - W * s_) / den if den > 0.2 else 0.0
    if L > 0 and sh.SizeWidth > 0:
        sh.Text.Story.Size = sh.Text.Story.Size * L / sh.SizeWidth
    # The polygon wraps the whole printed line, which is what SizeHeight measures, so it caps the
    # glyph size directly; the box-derived g is a loose stand-in used only when there is no polygon.
    cap_h = size_cap(lab, k, g)
    if cap_h > 0 and sh.SizeHeight > cap_h:
        sh.Text.Story.Size = sh.Text.Story.Size * cap_h / sh.SizeHeight


def thicken_text(sh, lab, k, col):
    """Faux bold: a same-colour round-joined outline (width from the measured source stroke)."""
    bp = float(lab.get('bold_px') or 0)
    if bp <= 0:
        return
    try:
        sh.Outline.Width = bp * k
        sh.Outline.Color.CopyAssign(col)
        sh.Outline.LineJoin = 1
    except Exception:  # noqa: BLE001 - cosmetic only, never fail the build
        pass


def draw_marks(app, doc, layer, marks, k, page_h):
    """Stipple dots and short hatch strokes PowerTRACE would drop, drawn as black vector shapes."""
    if not marks:
        return 0
    black = app.CreateColor(); black.RGBAssign(0, 0, 0)
    sr = app.CreateShapeRange()
    for cx, cy, r in marks.get('dots', []):
        sh = layer.CreateEllipse2(cx * k, page_h - cy * k, r * k, r * k)
        sh.Fill.ApplyUniformFill(black); sh.Outline.SetNoOutline()
        sr.Add(sh)
    for x0, y0, x1, y1, wd in marks.get('strokes', []):
        sh = layer.CreateLineSegment(x0 * k, page_h - y0 * k, x1 * k, page_h - y1 * k)
        sh.Outline.Width = wd * k
        sh.Outline.Color.CopyAssign(black)
        sh.Outline.LineCaps = 1                                   # round caps: short strokes stay blunt
        sr.Add(sh)
    n = int(sr.Count)
    if n > 1:
        try:
            sr.Group()
        except Exception:  # noqa: BLE001 - grouping is cosmetic
            pass
    return n


def draw_to_fill_marks(app, doc, page_w, page_h, labels, k):
    """Magenta outlines marking the labels that were left blank for a human to type.

    Those boxes were wiped from the image but no text object was placed, so the spot is simply empty -
    and an empty spot in a dense drawing is invisible. The outlines go on their own layer, are drawn
    AFTER result.png is exported (so the self-check still compares a clean page), and the layer is
    named so it can be deleted in one click once the typing is done.
    """
    todo = [l for l in (labels or []) if l.get('erase_only') or l.get('unmeasurable')]
    if not todo:
        return 0
    try:
        layer = doc.CreateLayer('TO FILL (delete after typing)')
    except Exception:  # noqa: BLE001 - older API surface: mark on the active layer instead
        layer = doc.ActiveLayer
    magenta = app.CreateColor(); magenta.RGBAssign(255, 0, 255)
    n = 0
    for lab in todo:
        try:
            x0, y0, x1, y1 = lab['box']
            w, h = (x1 - x0) * k, (y1 - y0) * k
            if w <= 0 or h <= 0:
                continue
            sh = layer.CreateRectangle2(x0 * k, page_h - y1 * k, w, h)
            try:
                sh.Fill.ApplyNoFill()
            except Exception:  # noqa: BLE001 - a filled rectangle would hide the drawing: drop it
                sh.Delete()
                continue
            sh.Outline.Color.CopyAssign(magenta)
            sh.Outline.Width = 0.015
            n += 1
        except Exception:  # noqa: BLE001 - the marks are a convenience, never a reason to fail a build
            pass
    return n


def export_page_png(app, doc, page_w, page_h, png_path, px_w, px_h):
    """Export exactly the page. CorelDRAW exports the extent of the objects, not the page, and squeezes
    it into SizeX x SizeY: empty margins stretched the drawing (+9 px) and one text object past the
    page edge shifted everything (up to 13 px). So: a page-sized white rectangle (no empty margins),
    export the whole extent at the page's pixel density, then crop the page out. The rectangle is
    deleted before the .cdr is saved."""
    import cv2
    import numpy as np
    bg = doc.ActiveLayer.CreateRectangle2(0, 0, page_w, page_h)
    white = app.CreateColor(); white.RGBAssign(255, 255, 255)
    bg.Fill.ApplyUniformFill(white); bg.Outline.SetNoOutline(); bg.OrderToBack()
    try:
        ext = doc.ActivePage.Shapes.All()
        left, bottom, ew, eh = ext.LeftX, ext.BottomY, ext.SizeWidth, ext.SizeHeight
        ppi_x, ppi_y = px_w / page_w, px_h / page_h
        eo = app.CreateStructExportOptions()
        eo.SizeX = int(round(ew * ppi_x)); eo.SizeY = int(round(eh * ppi_y))
        eo.AntiAliasingType = 1; eo.ImageType = 4  # 24-bit RGB, not 8-bit palette
        doc.Export(png_path, 802, 1, eo, None)
    finally:
        bg.Delete()
    img = cv2.imdecode(np.fromfile(png_path, np.uint8), cv2.IMREAD_COLOR)
    x0 = int(round((0 - left) * img.shape[1] / ew))
    y0 = int(round((bottom + eh - page_h) * img.shape[0] / eh))
    out = np.full((px_h, px_w, 3), 255, np.uint8)
    src = img[max(y0, 0):y0 + px_h, max(x0, 0):x0 + px_w]
    out[max(-y0, 0):max(-y0, 0) + src.shape[0], max(-x0, 0):max(-x0, 0) + src.shape[1]] = src
    cv2.imencode('.png', out)[1].tofile(png_path)


def draw_regions(app, doc, layer, shapes, k, page_h):
    """Flat-colour regions (cdr_flat) drawn directly as filled Bezier curves - no PowerTRACE. Shapes arrive
    in paint order (outer first); each is one closed path, optionally with the source's rim as outline.
    Returns (curves, nodes)."""
    made, nodes, outlined = [], 0, []
    cols = {}
    for s in shapes:
        paths = s.get('paths') or [{'start': s.get('start'), 'segs': s.get('segs') or []}]
        paths = [p_ for p_ in paths if p_.get('start') and len(p_.get('segs') or []) >= 2]
        if not paths:
            continue
        try:
            crv = app.CreateCurve(doc)
            for p_ in paths:                      # outer contour, then its holes (even-odd fill)
                x0, y0 = p_['start']
                sp = crv.CreateSubPath(x0 * k, page_h - y0 * k)
                for x, y, c1x, c1y, c2x, c2y in p_['segs']:
                    sp.AppendCurveSegment2(x * k, page_h - y * k, c1x * k, page_h - c1y * k,
                                           c2x * k, page_h - c2y * k)
                sp.Closed = True
            sh = layer.CreateCurve(crv)
        except Exception:  # noqa: BLE001 - one bad shape must not cost the drawing
            continue
        segs = [q for p_ in paths for q in p_['segs']]
        key = tuple(s['color'])
        if key not in cols:
            c = app.CreateColor(); c.RGBAssign(*[int(v) for v in s['color']]); cols[key] = c
        sh.Fill.ApplyUniformFill(cols[key])
        sh.Outline.SetNoOutline()
        if s.get('outline'):
            outlined.append((paths, s['outline']))
        made.append(sh)
        nodes += len(segs)
    # Outlines in a SECOND pass, on top of every fill. A frame that the model represents as a ring-shaped
    # region is the largest shape and is therefore painted first; with its outline attached, every later
    # fill covered the figure's own border, and only the element-level check noticed.
    for paths, ol in outlined:
        try:
            crv = app.CreateCurve(doc)
            for p_ in paths:
                x0, y0 = p_['start']
                sp = crv.CreateSubPath(x0 * k, page_h - y0 * k)
                for x, y, c1x, c1y, c2x, c2y in p_['segs']:
                    sp.AppendCurveSegment2(x * k, page_h - y * k, c1x * k, page_h - c1y * k,
                                           c2x * k, page_h - c2y * k)
                sp.Closed = True
            sh = layer.CreateCurve(crv)
            sh.Fill.ApplyNoFill()
            oc = app.CreateColor(); oc.RGBAssign(*[int(v) for v in ol['color']])
            sh.Outline.Color.CopyAssign(oc)                # colour BEFORE width (see patch_outline)
            sh.Outline.Width = max(float(ol['width_px']) * k, 0.001)
            sh.Outline.LineJoin = 1
            made.append(sh)
        except Exception:  # noqa: BLE001 - an outline is cosmetic, never fail the drawing over it
            continue
    if made:
        try:
            sr = app.CreateShapeRange()
            for sh in made:
                sr.Add(sh)
            sr.Group()
        except Exception:  # noqa: BLE001 - grouping is cosmetic
            pass
    return len(made), nodes


def _dash_style(app, dash_w, gap_w):
    """The built-in single-dash outline style closest to dash/gap (both in line widths, CorelDRAW's
    unit for dash patterns). A preset, not OutlineStyles.Add(): Add changes the user's application-wide
    style list, and COM offers no Remove to undo it."""
    import math as _m
    best, best_d = None, 1e9
    try:
        styles = app.OutlineStyles
        for i in range(1, int(styles.Count) + 1):
            s = styles.Item(i)
            if int(s.DashCount) != 1:
                continue
            d = abs(_m.log(float(s.DashLength(1)) / max(dash_w, 0.5))) + \
                abs(_m.log(float(s.GapLength(1)) / max(gap_w, 0.5)))
            if d < best_d:
                best, best_d = s, d
    except Exception:  # noqa: BLE001
        return None
    return best


def draw_rules(app, doc, layer, rules, k, page_h):
    """Graticule / grid rules as editable vector polylines with the source line width and grey."""
    if not rules or not rules.get('polylines'):
        return 0
    col = app.CreateColor(); col.RGBAssign(*[int(v) for v in rules['color']])
    width = float(rules['width_px']) * k
    style = _dash_style(app, rules['dash'][0] / float(rules['width_px']),
                        rules['dash'][1] / float(rules['width_px'])) if rules.get('dash') else None
    made = []
    for pl in rules['polylines']:
        pts = [(pl[i] * k, page_h - pl[i + 1] * k) for i in range(0, len(pl), 2)]
        if len(pts) < 2:
            continue
        try:
            crv = app.CreateCurve(doc)
            sp = crv.CreateSubPath(pts[0][0], pts[0][1])
            for x, y in pts[1:]:
                sp.AppendLineSegment(x, y)
            shapes = [layer.CreateCurve(crv)]
        except Exception:  # noqa: BLE001 - older API surface: fall back to plain segments
            shapes = [layer.CreateLineSegment(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
                      for i in range(len(pts) - 1)]
        for sh in shapes:
            sh.Outline.Width = width
            sh.Outline.Color.CopyAssign(col)
            if style is not None:
                try:
                    sh.Outline.Style = style
                except Exception:  # noqa: BLE001 - a solid line is still the right line
                    pass
            made.append(sh)
    if made:
        try:
            sr = app.CreateShapeRange()
            for sh in made:
                sr.Add(sh)
            sr.Group()
        except Exception:  # noqa: BLE001 - grouping is cosmetic
            pass
    return len(made)


def repair_lost_ink(app, doc, layer, work_dir, meta, k, page_h, S, gfx_png):
    """Redraw linework PowerTRACE dropped, as centerline strokes.

    PowerTRACE judges detail relative to the WHOLE bitmap: a 2 px line is 1.4% of a 140 px crop and
    survives, but 0.1% of a 2048 px map and is discarded as noise. That is how the orange band's black
    border (2 x 172 px, drawn in the source) vanished from the geology map - linework no hand edit can
    restore, because nothing in the file says it was ever there. So the ink layer is checked against
    what was actually drawn, and every thin run that is missing is put back the way the fault lines are
    drawn: a centreline stroke. Only long thin pieces qualify, so a dropped speck stays dropped.
    """
    try:
        import cv2
        import numpy as np

        import cdr_trace_skeleton as sk
    except Exception:  # noqa: BLE001 - repair is optional, never fail a build over it
        return 0
    ink_file = next((l['file'] for l in meta['layers'] if l['kind'] == 'ink'), None)
    if not ink_file:
        return 0
    try:
        rd = lambda p: cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_UNCHANGED)
        ink = rd(os.path.join(work_dir, ink_file.replace('/', os.sep))) < 128
        gfx = rd(gfx_png)
        if gfx is None or ink is None:
            return 0
        gfx = cv2.cvtColor(gfx[..., :3], cv2.COLOR_BGR2GRAY)
        gfx = cv2.resize(gfx, (ink.shape[1], ink.shape[0]), interpolation=cv2.INTER_AREA)
        drawn = cv2.dilate((gfx < 140).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        missing = (ink & ~drawn).astype(np.uint8)
        n, cc, st, _ = cv2.connectedComponentsWithStats(missing, 8)
        polys, widths = [], []
        for i in range(1, n):
            x, y, bw, bh, area = st[i]
            if max(bw, bh) < 20 * S or area < 24 * S:
                continue
            comp = (cc[y:y + bh, x:x + bw] == i).astype(np.uint8)
            skel = sk.skeletonize(comp * 255)
            ln = int(skel.sum())
            if ln < 10 * S or area / float(ln) > 4.0 * S:
                continue                       # not a thin run: leave it to the tracer
            for path in sk.trace_paths(skel):
                if len(path) < 6 * S:
                    continue
                ap = sk._simplify(path, 0.7 * S).reshape(-1, 2)
                polys.append([float(v) for xy in ap for v in (xy[0] + x, xy[1] + y)])
            widths.append(area / float(ln))
        if not polys:
            return 0
        rules = {'polylines': polys, 'width_px': round(max(float(np.median(widths)), 1.0), 2),
                 'color': [32, 32, 32]}
        return draw_rules(app, doc, layer, rules, k, page_h)
    except Exception:  # noqa: BLE001
        return 0


def _trace_mask_tiled(app, doc, layer, png_path, page_w, page_h, trace_type, detail, smoothing,
                      rgb, k, tile_px=None):
    """PowerTRACE the mask one TILE at a time, so thin lines are not judged against the whole page.

    The tracer's detail threshold is relative to the bitmap it is given: a 2 px line is 0.1% of a
    2048 px map and is discarded as noise, but 0.4% of a 500 px tile and survives. Tiles overlap by a
    few pixels so a line is not cut at a seam; the overlap is the same colour, so it does not show.
    Returns (group_or_None, (curves, nodes), tiles_used) and falls back to one whole-page trace when
    the mask is small enough that tiling would buy nothing.
    """
    import cv2
    import numpy as np

    tile_px = tile_px or int(os.environ.get('CDR_TILE_PX', '900') or 900)
    img = cv2.imdecode(np.fromfile(png_path, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, (0, 0), 0
    H, W = img.shape[:2]
    cols, rows = max(1, int(round(W / tile_px))), max(1, int(round(H / tile_px)))
    if cols * rows <= 1:
        return None, (0, 0), 0
    # A CLOSED ring cut by a seam becomes an open "C", and CorelDRAW fills an open path as if it were
    # closed: on a flowchart every box border that crossed a seam rendered as a solid black rectangle.
    # Such a layer is traced whole; repair_lost_ink still puts back thin lines the whole-page trace drops.
    seams_x = [int((c + 1) * W / cols) for c in range(cols - 1)]
    seams_y = [int((r + 1) * H / rows) for r in range(rows - 1)]
    ink = (img < 128).astype(np.uint8)
    cnts, hier = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is not None:
        for ci, hq in enumerate(hier[0]):
            if hq[3] >= 0 or cv2.contourArea(cnts[ci]) < 100:
                continue                          # only outer contours, and only real shapes
            holes = [q for q, h2 in enumerate(hier[0]) if h2[3] == ci and cv2.contourArea(cnts[q]) >= 50]
            if not holes:
                continue                          # not a ring: nothing to be cut open
            x, y, bw, bh = cv2.boundingRect(cnts[ci])
            if any(x < sx < x + bw for sx in seams_x) or any(y < sy < y + bh for sy in seams_y):
                return None, (0, 0), 0
    ov = 6
    made, curves, nodes, used = [], 0, 0, 0
    base = os.path.splitext(png_path)[0]
    for r in range(rows):
        for c in range(cols):
            x0, x1 = max(int(c * W / cols) - ov, 0), min(int((c + 1) * W / cols) + ov, W)
            y0, y1 = max(int(r * H / rows) - ov, 0), min(int((r + 1) * H / rows) + ov, H)
            tile = img[y0:y1, x0:x1]
            if tile.size == 0 or not (tile < 128).any():
                continue                              # nothing drawn in this tile
            tp = f'{base}_tile{r}{c}.png'
            cv2.imencode('.png', tile)[1].tofile(tp)
            try:
                layer.Import(tp, 0, app.CreateStructImportOptions())
                bmp = doc.ActiveShape
                bmp.SetSize((x1 - x0) * page_w / W, (y1 - y0) * page_h / H)
                bmp.LeftX = x0 * page_w / W
                bmp.BottomY = page_h - y1 * page_h / H
                ts = bmp.Bitmap.Trace(TRACE_TYPES.get(trace_type, 1), smoothing, detail, 8, 0, 2,
                                      True, True, True)
                ts.DetailLevelPercent = detail; ts.Smoothing = smoothing; ts.CornerSmoothness = 0
                ts.MergeAdjacentObjects = True
                ts.ApplyChanges()
                curves += int(ts.CurveCount); nodes += int(ts.NodeCount)
                ts.Finish()
                grp = doc.ActiveShape
                if rgb is not None:
                    col = app.CreateColor(); col.RGBAssign(*[int(v) for v in rgb])
                    for sh in group_children(grp):
                        try:
                            sh.Fill.ApplyUniformFill(col)
                            sh.Outline.SetNoOutline()
                        except Exception:  # noqa: BLE001
                            pass
                made.append(grp)
                used += 1
            except Exception:  # noqa: BLE001 - one bad tile must not lose the layer
                pass
            finally:
                try:
                    os.remove(tp)
                except OSError:
                    pass
    if not made:
        return None, (0, 0), 0
    try:
        sr = app.CreateShapeRange()
        for g in made:
            sr.Add(g)
        return sr.Group(), (curves, nodes), used
    except Exception:  # noqa: BLE001 - grouping is cosmetic
        return made[0], (curves, nodes), used


def build_color(app, doc, work_dir, font, trace_type, detail, smoothing, fonts=()):
    """Multi-layer colour build: one PowerTRACE per palette colour, ink layer last, then text."""
    meta = json.load(open(os.path.join(work_dir, 'layers.json'), encoding='utf-8'))
    w, h = meta['src_size']; S = meta['scale']
    page_w, page_h = w / 72.0, h / 72.0
    k = page_w / (w * S)
    doc.Unit = 3
    doc.ActivePage.SetSize(page_w, page_h)
    layer = doc.ActiveLayer
    done = []
    rules_drawn = False
    for lay in meta['layers']:
        if not rules_drawn and lay['kind'] in ('ink', 'ink_faint'):
            done.append({'layer': 'rules', 'kind': 'rules', 'lines': draw_rules(app, doc, layer, meta.get('rules'), k, page_h)})
            rules_drawn = True
        if lay['kind'] == 'regions':
            curves, nodes = draw_regions(app, doc, layer, lay.get('shapes') or [], k, page_h)
            done.append({'layer': 'regions', 'kind': 'regions', 'curves': curves, 'nodes': nodes})
            continue
        if lay.get('strokes'):
            # a layer of thin lines: drawn as centerline strokes like the rules, not outline-traced
            done.append({'layer': os.path.basename(lay['file']), 'kind': 'strokes', 'color': lay['color'],
                         'lines': draw_rules(app, doc, layer, lay['strokes'], k, page_h),
                         **({'dash': lay['strokes']['dash']} if lay['strokes'].get('dash') else {})})
            if lay.get('rest_file'):
                # the pieces of a dashed layer no line explained: traced like any colour layer
                png = os.path.join(work_dir, lay['rest_file'].replace('/', os.sep))
                _, (curves, nodes) = _trace_mask(app, doc, layer, png, page_w, page_h,
                                                 trace_type, detail, smoothing, lay['color'], None, k)
                done.append({'layer': os.path.basename(png), 'kind': 'color', 'color': lay['color'],
                             'curves': curves, 'nodes': nodes})
            continue
        png = os.path.join(work_dir, lay['file'].replace('/', os.sep))
        rgb = lay['color'] if lay['kind'] == 'color' else lay['color']
        tiles = 0
        if TILE_TRACE and lay['kind'] in ('ink', 'ink_faint') and not lay.get('outline'):
            # the linework layer is where the whole-page detail threshold costs real lines
            _g, (curves, nodes), tiles = _trace_mask_tiled(app, doc, layer, png, page_w, page_h,
                                                           trace_type, detail, smoothing, rgb, k)
        if not tiles:
            _, (curves, nodes) = _trace_mask(app, doc, layer, png, page_w, page_h,
                                             trace_type, detail, smoothing, rgb, lay.get('outline'), k)
        done.append({'layer': os.path.basename(png), 'kind': lay['kind'], 'color': lay['color'],
                     'curves': curves, 'nodes': nodes, **({'tiles': tiles} if tiles else {})})
    if not rules_drawn:
        done.append({'layer': 'rules', 'kind': 'rules', 'lines': draw_rules(app, doc, layer, meta.get('rules'), k, page_h)})
    # CHECKPOINT: graphics-only export before the labels go on (see build() for why)
    gfx = os.path.join(work_dir, 'graphics_only.png')
    export_page_png(app, doc, page_w, page_h, gfx, w * 2, h * 2)
    lost = repair_lost_ink(app, doc, layer, work_dir, meta, k, page_h, S, gfx)
    if lost:
        done.append({'layer': 'ink_repair', 'kind': 'strokes', 'color': [32, 32, 32], 'lines': lost})
        export_page_png(app, doc, page_w, page_h, gfx, w * 2, h * 2)
    fits = []
    for lab in meta['labels']:
        if lab.get('erase_only') or lab.get('unmeasurable'):
            continue          # left blank for a human; marked on the TO FILL layer below
        x0, y0, x1, y1 = lab['tight']
        bw = (x1 - x0) * k
        cx, cy = (x0 + x1) / 2 * k, page_h - (y0 + y1) / 2 * k
        # the measured run centre anchors the label better than the box centre when the box is off
        # (slanted labels only, like the sizing below: on a horizontal label the tight box is exact, while
        # the run centre was pulled onto an adjoining frame line ("39°" landed on the map border) or
        # lifted by a glyph clipped at the image edge ("75°" printed half off the page))
        meas = lab.get('measured') or {}
        # Only for a real slant. At 5-7 degrees the box centre is still the better anchor, while the
        # measured run centre is pulled sideways by whatever shares the box - the map frame next to
        # "39°" clipped its first digit off the page.
        if (meas.get('cx') is not None and meas.get('cy') is not None
                and 15 <= abs(float(lab.get('angle') or 0.0)) < 75):
            cx, cy = float(meas['cx']) * k, page_h - float(meas['cy']) * k
        sh = create_label(layer, lab, cx, cy, font, meta, fonts, k, S)
        rgb = lab.get('color') or [0, 0, 0]
        col = app.CreateColor(); col.RGBAssign(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        sh.Fill.ApplyUniformFill(col)
        ang = float(lab.get('angle') or 0.0)
        gh = float(lab.get('glyph_h') or 0) * k          # measured glyph height of the original label
        # measured runs size slanted labels only; horizontal ones keep the tuned box/text_h fit
        if not (MEASURED_TEXT and 4 < abs(ang) < 75 and fit_measured_text(sh, lab, k)):
            # Fallback: the box-derived fits below, used when the label has no measured run to size from
            # (see MEASURED_TEXT above) or when the measurement is missing/unusable.
            if abs(ang) >= 75:
                fit_rotated_text(sh, lab, k)             # vertical label (axis title)
            elif abs(ang) > 4:
                # slanted label: OCR box width is the diagonal span; fit by height instead
                if gh > 0:
                    sh.Text.Story.Size = sh.Text.Story.Size * gh / sh.SizeHeight
            elif lab.get('text_h'):
                fit_horizontal_text(sh, lab, k, S)
            else:
                sh.Text.Story.Size = sh.Text.Story.Size * bw / sh.SizeWidth
                # CJK glyphs split into radicals, so the median component height underestimates the
                # glyph size; never cap below ~the confirmed row height
                if 'box' in lab:
                    gh = max(gh, 0.7 * (lab['box'][3] - lab['box'][1]) * k)
                if gh > 0 and sh.SizeHeight > 1.25 * gh:
                    sh.Text.Story.Size = sh.Text.Story.Size * 1.25 * gh / sh.SizeHeight   # width fit over-shot
        try:
            # observability: the size each label ended up at, and the cap that was available. Sizing
            # bugs are invisible once the object is rotated (SizeHeight then reads the rotated box).
            fits.append({'text': lab.get('text', '')[:24], 'pt': round(float(sh.Text.Story.Size), 1),
                         'line_px': round(float(sh.SizeHeight) / k, 1),
                         'cap_px': round(size_cap(lab, k) / k, 1) if size_cap(lab, k) else None})
        except Exception:  # noqa: BLE001
            pass
        thicken_text(sh, lab, k, col)
        sh.CenterX = cx; sh.CenterY = cy
        if abs(ang) > 4:
            sh.Rotate(-ang)                              # image-space slant -> page-space (y up) rotation
            sh.CenterX = cx; sh.CenterY = cy
    # only the labels the cap actually held back are worth reporting; the full list is noise
    capped = [f for f in fits if f.get('cap_px') and f['line_px'] >= 0.98 * f['cap_px']][:20]
    return {'layers_built': done, 'labels': len(meta['labels']), 'page_in': [page_w, page_h],
            'text_size_capped': capped}


def build(work_dir, font=None, trace_type='lineart', detail=100, smoothing=25, mode='lineart'):
    if mode == 'color':
        meta = json.load(open(os.path.join(work_dir, 'layers.json'), encoding='utf-8'))
    else:
        meta = json.load(open(os.path.join(work_dir, 'labels.json'), encoding='utf-8'))
    w, h = meta['src_size']; S = meta['scale']
    page_w, page_h = w / 72.0, h / 72.0          # 1 source px = 1 pt
    k = page_w / (w * S)                         # working px -> inch
    import win32com.client
    app = win32com.client.Dispatch('CorelDRAW.Application')
    # A just-started CorelDRAW answers Documents.Count but not yet FontList ("does not support
    # enumeration"), and that crashed the build seconds after an automatic restart. The list is only
    # used to check a font exists, so it is worth waiting briefly for - and doing without if it never
    # arrives, in which case the requested font is taken at face value.
    fonts = set()
    for _try in range(10):
        try:
            fonts = {str(f) for f in app.FontList}
            if fonts:
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
    if fonts and (not font or font not in fonts or font == '宋体'):
        font = next((f for f in FONT_PREF if f in fonts), font or FONT_PREF[0])
    elif not font:
        font = FONT_PREF[0]
    cdr_path = os.path.join(work_dir, 'result.cdr'); png_path = os.path.join(work_dir, 'result.png')
    # A build that was killed mid-COM (the step's own timeout does exactly that) never reached the
    # close below, so its half-built document is still open here. Left alone they pile up and every
    # later build in the same instance gets slower. Close whatever is open before starting a new one.
    try:
        for _ in range(int(app.Documents.Count)):
            d = app.ActiveDocument
            d.Dirty = False
            d.Close()
    except Exception:  # noqa: BLE001 - nothing open, or an instance that will be restarted anyway
        pass
    doc = None; t0 = time.time(); info = {}
    try:
        doc = None
        for _try in range(10):
            # A restarted CorelDRAW exposes Documents and the font list before it exposes the
            # document API: the call below then fails with "Application.CreateDocumentEx". Wait for
            # it rather than losing the build to a cold start.
            try:
                doc = app.CreateDocumentEx(app.CreateStructCreateOptions())
                break
            except Exception:  # noqa: BLE001
                try:
                    doc = app.CreateDocument()
                    break
                except Exception:  # noqa: BLE001
                    time.sleep(3)
        if doc is None:
            raise RuntimeError('CorelDRAW did not expose its document API (cold start?)')
        if mode == 'color':
            info.update(build_color(app, doc, work_dir, font, trace_type, detail, smoothing, fonts))
            export_page_png(app, doc, page_w, page_h, png_path, w * 2, h * 2)
            # drawn after the export so result.png - and the self-check that reads it - stays clean
            info['to_fill_marks'] = draw_to_fill_marks(app, doc, page_w, page_h, meta['labels'], k)
            doc.SaveAs(cdr_path, app.CreateStructSaveAsOptions())
            info.update(ok=True, cdr=cdr_path, png=png_path, font=font, mode='color',
                        total_seconds=round(time.time() - t0, 1))
            return info
        doc.Unit = 3  # cdrInch
        doc.ActivePage.SetSize(page_w, page_h)
        layer = doc.ActiveLayer
        # tracing a 2x-upscaled copy was tried: small drawing better, clean drawing worse and 2x slower,
        # stray diagonals appeared. Thin strokes are thickened in finalize() instead.
        trace_png = os.path.join(work_dir, 'sr2_no_text.png')
        layer.Import(trace_png, 0, app.CreateStructImportOptions())
        bmp = doc.ActiveShape
        bmp.SetSize(page_w, page_h); bmp.LeftX = 0; bmp.BottomY = 0
        ts = bmp.Bitmap.Trace(TRACE_TYPES.get(trace_type, 1), smoothing, detail, 8, 0, 2, True, False, False)
        ts.DetailLevelPercent = detail; ts.Smoothing = smoothing; ts.CornerSmoothness = 0; ts.MergeAdjacentObjects = True
        ts.ApplyChanges()
        info.update(curves=int(ts.CurveCount), nodes=int(ts.NodeCount), trace_seconds=round(time.time() - t0, 1))
        ts.Finish()
        # CHECKPOINT: export right after the trace, BEFORE the fine marks are drawn and before any
        # text, so it matches sr2_no_text.png one to one - the exact image PowerTRACE was handed,
        # which holds neither text nor fine marks (extract_fine_marks took those out). Comparing
        # after draw_marks would report the deliberately redrawn dots as "extra graphics".
        export_page_png(app, doc, page_w, page_h, os.path.join(work_dir, 'graphics_only.png'), w * 2, h * 2)
        info['fine_marks'] = draw_marks(app, doc, layer, meta.get('marks'), k, page_h)
        black = app.CreateColor(); black.RGBAssign(0, 0, 0)
        for lab in meta['labels']:
            if lab.get('erase_only'):
                continue      # left blank for a human; marked on the TO FILL layer below
            x0, y0, x1, y1 = lab['tight']
            bw = (x1 - x0) * k
            cx, cy = (x0 + x1) / 2 * k, page_h - (y0 + y1) / 2 * k
            s = create_label(layer, lab, cx, cy, font, meta, fonts, k, S)
            s.Fill.ApplyUniformFill(black)
            ang = float(lab.get('angle') or 0.0)
            if abs(ang) > 4:
                fit_rotated_text(s, lab, k)
                s.Rotate(-ang)                           # image-space angle -> page-space rotation
            else:
                fit_horizontal_text(s, lab, k, S)
                thicken_text(s, lab, k, black)
            s.CenterX = cx; s.CenterY = cy
        export_page_png(app, doc, page_w, page_h, png_path, w * 2, h * 2)
        # drawn after the export so result.png - and the self-check that reads it - stays clean
        info['to_fill_marks'] = draw_to_fill_marks(app, doc, page_w, page_h, meta['labels'], k)
        doc.SaveAs(cdr_path, app.CreateStructSaveAsOptions())
        info.update(ok=True, cdr=cdr_path, png=png_path, font=font, labels=len(meta['labels']),
                    total_seconds=round(time.time() - t0, 1))
    finally:
        if doc is not None:
            try:
                doc.Dirty = False; doc.Close()
            except Exception:  # noqa: BLE001
                pass
    return info


def time_budget(work_dir, mode):
    """Seconds the CorelDRAW step may take. A fixed 240 s killed a dense chart (8 colour layers, 260 rules,
    30 labels) mid-build; a hung modal dialog should still fail in reasonable time. Measured cost drivers:
    ~20-30 s per traced layer at a few Mpx, a few ms per vector rule/mark, ~0.5 s per label."""
    name = 'layers.json' if mode == 'color' else 'labels.json'
    try:
        meta = json.load(open(os.path.join(work_dir, name), encoding='utf-8'))
    except Exception:  # noqa: BLE001
        return 240
    layers = meta.get('layers') or [None]
    # every polyline drawn through COM costs about the same, whether it is a graticule rule or a
    # colour layer drawn as centreline strokes - and the stroke layers were missing from this sum,
    # which is how a grid map with 452 strokes in one layer (25 s of real work) ran out of budget
    rules = len((meta.get('rules') or {}).get('polylines', []))
    for lay in layers:
        st = (lay or {}).get('strokes') or {}
        rules += len(st.get('polylines', []) if isinstance(st, dict) else st)
    marks = meta.get('marks') or {}
    n_marks = len(marks.get('dots', [])) + len(marks.get('strokes', []))
    # a tiled ink layer is one import+trace per tile instead of one for the page
    tiles = 0
    if TILE_TRACE:
        try:
            tw, th = meta['src_size']; S = meta.get('scale') or 1
            tp = int(os.environ.get('CDR_TILE_PX', '900') or 900)
            n = max(1, int(round(tw * S / tp))) * max(1, int(round(th * S / tp)))
            tiles = (n if n > 1 else 0) * sum(
                1 for lay in layers if (lay or {}).get('kind') in ('ink', 'ink_faint')
                and not (lay or {}).get('outline') and not (lay or {}).get('strokes'))
        except Exception:  # noqa: BLE001
            tiles = 0
    budget = (120 + 30 * len(layers) + 0.5 * len(meta.get('labels', []))
              + 0.1 * rules + 0.02 * n_marks + 4 * tiles)
    return int(min(max(budget, 240), 900))


def main():
    args = sys.argv[1:]
    work_dir = os.path.abspath(args[0])     # CorelDRAW resolves relative paths against its own cwd
    font = args[1] if len(args) > 1 and args[1] else None
    trace_type = args[2] if len(args) > 2 else 'lineart'
    detail = int(args[3]) if len(args) > 3 else 100
    smoothing = int(args[4]) if len(args) > 4 else 25
    mode = args[5] if len(args) > 5 else 'lineart'
    result = {}

    def work():
        pythoncom.CoInitialize()
        try:
            result.update(build(work_dir, font, trace_type, detail, smoothing, mode))
        except Exception as exc:  # noqa: BLE001
            result.update(ok=False, error=f'{type(exc).__name__}: {exc}', trace=traceback.format_exc()[-800:])

    t = threading.Thread(target=work, daemon=True); t.start(); t.join(time_budget(work_dir, mode))
    if t.is_alive():
        print(json.dumps({'ok': False, 'error': 'CorelDRAW COM call timed out (modal dialog open?)'}, ensure_ascii=False))
        sys.stdout.flush(); os._exit(3)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
