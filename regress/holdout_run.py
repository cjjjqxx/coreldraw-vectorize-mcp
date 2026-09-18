"""Run the one-call MCP on unseen images (no ground truth, no tuning) and collect what it reports.

usage: python holdout_run.py <run_id> [image_name ...]   (images from regress/holdout/*.png)
writes holdout_runs/<run_id>/<name>/ (work dir), summary.json and compare_<name>.png
"""
import glob
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def sheet(src_path, res_path, diff_path, out_path, width=900):
    ims = [imread(src_path), imread(res_path)]
    if diff_path and os.path.exists(diff_path):
        ims.append(imread(diff_path))
    h, w = ims[0].shape[:2]
    rows = []
    for im in ims:
        im = cv2.resize(im, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        rows += [im, np.full((6, width, 3), (0, 0, 255), np.uint8)]
    cv2.imencode('.png', np.vstack(rows))[1].tofile(str(out_path))


def main(run_id, names=None):
    import cdr_server
    run = HERE / 'holdout_runs' / run_id
    run.mkdir(parents=True, exist_ok=True)
    imgs = sorted(glob.glob(str(HERE / 'holdout' / '*.png')))
    if names:
        imgs = [p for p in imgs if Path(p).stem in names]
    summary_path = run / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
    for p in imgs:
        name = Path(p).stem
        work = run / name
        print(f'[{time.strftime("%H:%M:%S")}] {name}', flush=True)
        t0 = time.time()
        try:
            r = cdr_server.cdr_vectorize(p, str(work))
        except Exception as exc:  # noqa: BLE001
            r = {'ok': False, 'error': f'{type(exc).__name__}: {exc}', 'trace': traceback.format_exc()[-600:]}
        rec = {'seconds': round(time.time() - t0), 'ok': r.get('ok'), 'error': r.get('error'),
               'mode': r.get('mode'), 'labels': len(r.get('labels', [])),
               'to_check': len(r.get('labels_to_check', [])), 'unlabeled': len(r.get('unlabeled_text', [])),
               'issues': [i.get('type') for i in r.get('issues', [])], 'self_check': r.get('self_check'),
               'needs_review': r.get('needs_review')}
        (work / 'mcp_result.json').write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding='utf-8') \
            if work.exists() else None
        if r.get('ok'):
            sheet(work / 'src.png', r['png'], (r.get('self_check') or {}).get('diff_png'), run / f'compare_{name}.png')
        summary[name] = rec
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
        print('   ', json.dumps(rec, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:] or None)
