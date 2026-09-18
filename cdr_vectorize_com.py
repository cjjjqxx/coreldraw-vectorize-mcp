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
TRACE_TYPES = {'lineart': 1, 'logo': 2, 'detailed_logo': 3, 'technical': 7, 'line_drawing': 8}



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
    latin = meta.get('latin_font')
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
    if g > 0 and sh.SizeHeight > 1.3 * g:
        sh.Text.Story.Size = sh.Text.Story.Size * 1.3 * g / sh.SizeHeight


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


def draw_rules(app, doc, layer, rules, k, page_h):
    """Graticule / grid rules as editable vector polylines with the source line width and grey."""
    if not rules or not rules.get('polylines'):
        return 0
    col = app.CreateColor(); col.RGBAssign(*[int(v) for v in rules['color']])
    width = float(rules['width_px']) * k
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
        png = os.path.join(work_dir, lay['file'].replace('/', os.sep))
        rgb = lay['color'] if lay['kind'] == 'color' else lay['color']
        _, (curves, nodes) = _trace_mask(app, doc, layer, png, page_w, page_h,
                                         trace_type, detail, smoothing, rgb, lay.get('outline'), k)
        done.append({'layer': os.path.basename(png), 'kind': lay['kind'], 'color': lay['color'],
                     'curves': curves, 'nodes': nodes})
    if not rules_drawn:
        done.append({'layer': 'rules', 'kind': 'rules', 'lines': draw_rules(app, doc, layer, meta.get('rules'), k, page_h)})
    for lab in meta['labels']:
        x0, y0, x1, y1 = lab['tight']
        bw = (x1 - x0) * k
        cx, cy = (x0 + x1) / 2 * k, page_h - (y0 + y1) / 2 * k
        sh = create_label(layer, lab, cx, cy, font, meta, fonts, k, S)
        rgb = lab.get('color') or [0, 0, 0]
        col = app.CreateColor(); col.RGBAssign(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        sh.Fill.ApplyUniformFill(col)
        ang = float(lab.get('angle') or 0.0)
        gh = float(lab.get('glyph_h') or 0) * k          # measured glyph height of the original label
        if abs(ang) >= 75:
            fit_rotated_text(sh, lab, k)                 # vertical label (axis title)
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
        thicken_text(sh, lab, k, col)
        sh.CenterX = cx; sh.CenterY = cy
        if abs(ang) > 4:
            sh.Rotate(-ang)                              # image-space slant -> page-space (y up) rotation
            sh.CenterX = cx; sh.CenterY = cy
    return {'layers_built': done, 'labels': len(meta['labels']), 'page_in': [page_w, page_h]}


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
    fonts = {str(f) for f in app.FontList}
    if not font or font not in fonts or font == '宋体':
        font = next(f for f in FONT_PREF if f in fonts)
    cdr_path = os.path.join(work_dir, 'result.cdr'); png_path = os.path.join(work_dir, 'result.png')
    doc = None; t0 = time.time(); info = {}
    try:
        doc = app.CreateDocumentEx(app.CreateStructCreateOptions())
        if mode == 'color':
            info.update(build_color(app, doc, work_dir, font, trace_type, detail, smoothing, fonts))
            export_page_png(app, doc, page_w, page_h, png_path, w * 2, h * 2)
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
        info['fine_marks'] = draw_marks(app, doc, layer, meta.get('marks'), k, page_h)
        black = app.CreateColor(); black.RGBAssign(0, 0, 0)
        for lab in meta['labels']:
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
    rules = len((meta.get('rules') or {}).get('polylines', []))
    marks = meta.get('marks') or {}
    n_marks = len(marks.get('dots', [])) + len(marks.get('strokes', []))
    budget = (120 + 30 * len(meta.get('layers', [None])) + 0.5 * len(meta.get('labels', []))
              + 0.05 * rules + 0.02 * n_marks)
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
