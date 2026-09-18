"""Side-by-side comparison sheets for a run vs the baseline run.

usage: python make_compare.py <run_id> [baseline_id]
writes runs/<run_id>/compare_<case>.png (full view + zoom) and runs/<run_id>/compare_all.png
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FONT = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 22)
ZOOMS = {'tibet': (330, 170, 620, 330), 'well': (60, 60, 700, 250)}      # reference px


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def titled(img, title, w):
    h = int(img.shape[0] * w / img.shape[1])
    im = Image.fromarray(cv2.cvtColor(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    canvas = Image.new('RGB', (w, h + 34), 'white')
    canvas.paste(im, (0, 34))
    ImageDraw.Draw(canvas).text((6, 4), title, fill=(180, 0, 0), font=FONT)
    return canvas


def row(images, titles, w):
    tiles = [titled(i, t, w) for i, t in zip(images, titles)]
    h = max(t.height for t in tiles)
    out = Image.new('RGB', (sum(t.width for t in tiles) + 8 * (len(tiles) - 1), h), (200, 200, 200))
    x = 0
    for t in tiles:
        out.paste(t, (x, 0)); x += t.width + 8
    return out


def main(run_id, base_id='baseline'):
    cases = {c['id']: c for c in json.load(open(HERE / 'cases.json', encoding='utf-8'))}
    run, base = HERE / 'runs' / run_id, HERE / 'runs' / base_id
    sums = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    bsum = json.loads((base / 'summary.json').read_text(encoding='utf-8'))
    sheets = []
    for cid, case in cases.items():
        if cid not in sums or not (run / cid / 'work' / 'result.png').exists():
            continue                                   # partial run
        ref = imread(HERE / 'refs' / case['ref'])
        H, W = ref.shape[:2]
        inp = cv2.resize(imread(run / cid / 'input.png'), (W, H), interpolation=cv2.INTER_NEAREST)
        a = cv2.resize(imread(base / cid / 'work' / 'result.png'), (W * 2, H * 2), interpolation=cv2.INTER_AREA)
        b = cv2.resize(imread(run / cid / 'work' / 'result.png'), (W * 2, H * 2), interpolation=cv2.INTER_AREA)
        ma, mb = bsum[cid].get('metrics', {}), sums[cid].get('metrics', {})
        fmt = lambda m: f"文字{m.get('text_ok', 0):.0%}"
        titles = ['干净原图', f'输入: {case.get("degrade") or "原图"}', f'基线输出 ({fmt(ma)})', f'现在输出 ({fmt(mb)})']
        full = row([ref, inp, a, b], titles, 560)
        zx0, zy0, zx1, zy1 = ZOOMS['tibet' if cid.startswith('tibet') else 'well']
        crop = lambda im, s: im[zy0 * s:zy1 * s, zx0 * s:zx1 * s]
        up = lambda im: cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        zoom = row([up(crop(ref, 1)), up(crop(inp, 1)), crop(a, 2), crop(b, 2)],
                   ['局部 原图', '局部 输入', '局部 基线', '局部 现在'], 560)
        sheet = Image.new('RGB', (full.width, full.height + zoom.height + 50), 'white')
        ImageDraw.Draw(sheet).text((8, 8), f'用例 {cid}', fill=(0, 0, 0), font=FONT)
        sheet.paste(full, (0, 44)); sheet.paste(zoom, (0, full.height + 50))
        sheet.save(run / f'compare_{cid}.png')
        sheets.append(sheet)
    print('\n'.join(str(run / f'compare_{c}.png') for c in cases))


if __name__ == '__main__':
    main(*sys.argv[1:3])
