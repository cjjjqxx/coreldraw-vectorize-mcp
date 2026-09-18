"""Regression harness for the image -> CDR vectorize MCP.

Every case = a clean reference image + a degradation (resize / blur / JPEG / grey). The degraded
image goes through the real MCP tool functions (cdr_server.cdr_vectorize_prepare / _build) with NO
manual label review - exactly what a calling AI gets from one call - and the result is scored
against the clean reference (score.py). A change to the MCP is accepted only if no case regresses.

usage:
  python vec_regress.py run [run_id] [case_id ...]     run cases (all by default)
  python vec_regress.py rescore <run_id>                re-score an existing run
  python vec_regress.py compare <run_a> <run_b>        metric deltas, regressions flagged
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import score as scorer  # noqa: E402

# metric -> +1 higher is better, -1 lower is better; tolerance before a change counts as regression
METRICS = {'color_f1': (1, 0.02), 'color_dE': (-1, 1.5), 'ink_f1': (1, 0.02), 'grid_f1': (1, 0.04),
           'extra_ink_blobs': (-1, 2), 'lost_ink_symbols': (-1, 2), 'lost_color_patches': (-1, 2),
           'text_ok': (1, 0.03), 'size_err_median': (-1, 0.05)}


def degrade(img, ops):
    for op in ops:
        kind = op[0]
        if kind == 'resize':
            f = float(op[1])
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        elif kind == 'blur':
            img = cv2.GaussianBlur(img, (0, 0), float(op[1]))
        elif kind == 'jpeg':
            ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, int(op[1])])
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        elif kind == 'gray':
            img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError(f'unknown degradation {op}')
    return img


ROMAN = re.compile(r'^[IVXLTAl1|]+$')


def auto_labels(candidates):
    """BASELINE label selection with no review (to be moved into the MCP): keep confident readings,
    drop roman-numeral-like tokens and single non-CJK characters (they stay linework)."""
    out = []
    for c in candidates:
        text = str(c['text']).strip()
        best = max((r[1] for r in c.get('readings', [])), default=0.0)
        if not text or best < 0.5:
            continue
        if ROMAN.match(text.replace(' ', '')):
            continue
        if len(text) == 1 and not re.match(r'[一-鿿]', text):
            continue
        out.append({'text': text, 'box': c['box'], 'angle': c.get('angle', 0.0)})
    return out


def run_case(case, run_dir):
    import cdr_server
    cdir = run_dir / case['id']
    work = cdir / 'work'
    work.mkdir(parents=True, exist_ok=True)
    ref_path = HERE / 'refs' / case['ref']
    ref = scorer.imread(str(ref_path))
    inp = cdir / 'input.png'
    cv2.imencode('.png', degrade(ref, case.get('degrade', [])))[1].tofile(str(inp))
    rec = {'id': case['id'], 'mode': case['mode'], 'degrade': case.get('degrade', [])}
    t0 = time.time()
    res = cdr_server.cdr_vectorize(str(inp), str(work))          # the one-call MCP flow, no review
    rec['seconds'] = round(time.time() - t0, 1)
    (cdir / 'mcp_result.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    if not res.get('ok'):
        rec['error'] = {'step': res.get('step', 'mcp'), 'error': str(res.get('error'))[-600:]}
        return rec
    rec['mode_detected'] = res.get('mode')
    rec['labels_used'] = len(res.get('labels', []))
    rec['to_check'] = len(res.get('labels_to_check', []))
    rec['unlabeled'] = len(res.get('unlabeled_text', []))
    build = res
    rep = scorer.score(str(ref_path), build['png'], str(HERE / 'gt' / case['gt']), str(cdir))
    rec['metrics'] = rep['metrics']
    return rec


def run(run_id=None, only=None):
    cases = json.load(open(HERE / 'cases.json', encoding='utf-8'))
    if only:
        cases = [c for c in cases if c['id'] in only]
    run_id = run_id or time.strftime('%m%d_%H%M')
    run_dir = HERE / 'runs' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / 'summary.json'
    results = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
    for case in cases:
        print(f'[{time.strftime("%H:%M:%S")}] {case["id"]} ...', flush=True)
        try:
            rec = run_case(case, run_dir)
        except Exception as exc:  # noqa: BLE001
            rec = {'id': case['id'], 'error': {'step': 'harness', 'error': f'{type(exc).__name__}: {exc}',
                                               'trace': traceback.format_exc()[-800:]}}
        results[case['id']] = rec
        summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
        print('   ', json.dumps(rec.get('metrics') or rec.get('error'), ensure_ascii=False), flush=True)
    write_table(run_dir, results)
    return results


def rescore(run_id):
    """Re-score an existing run (after a scorer / ground-truth change) without re-running the MCP."""
    cases = {c['id']: c for c in json.load(open(HERE / 'cases.json', encoding='utf-8'))}
    run_dir = HERE / 'runs' / run_id
    results = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    for cid, rec in results.items():
        png = run_dir / cid / 'work' / 'result.png'
        if cid in cases and png.exists() and rec.get('metrics'):
            case = cases[cid]
            rec['metrics'] = scorer.score(str(HERE / 'refs' / case['ref']), str(png),
                                          str(HERE / 'gt' / case['gt']), str(run_dir / cid))['metrics']
    (run_dir / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
    write_table(run_dir, results)


def write_table(run_dir, results):
    cols = list(METRICS)
    lines = ['| case | ' + ' | '.join(cols) + ' |', '|' + '---|' * (len(cols) + 1)]
    for cid, rec in results.items():
        m = rec.get('metrics')
        if not m:
            lines.append(f'| {cid} | ERROR {rec.get("error", {}).get("step")} |')
            continue
        lines.append(f'| {cid} | ' + ' | '.join('' if m.get(c) is None else str(m.get(c)) for c in cols) + ' |')
    (run_dir / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


def compare(run_a, run_b):
    a = json.loads((HERE / 'runs' / run_a / 'summary.json').read_text(encoding='utf-8'))
    b = json.loads((HERE / 'runs' / run_b / 'summary.json').read_text(encoding='utf-8'))
    regressions = []
    for cid in b:
        ma, mb = (a.get(cid) or {}).get('metrics'), b[cid].get('metrics')
        if not ma or not mb:
            print(f'{cid}: missing metrics in one run')
            continue
        parts = []
        for k, (sign, tol) in METRICS.items():
            va, vb = ma.get(k), mb.get(k)
            if va is None or vb is None:
                continue
            d = vb - va
            flag = ''
            if sign * d < -tol:
                flag = ' REGRESSION'
                regressions.append((cid, k, va, vb))
            elif sign * d > tol:
                flag = ' better'
            if flag:
                parts.append(f'{k} {va}->{vb}{flag}')
        print(f'{cid}: ' + ('; '.join(parts) if parts else 'no significant change'))
    print(f'\n{len(regressions)} regression(s)')
    return regressions


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'run':
        run(sys.argv[2] if len(sys.argv) > 2 else None, sys.argv[3:] or None)
    elif len(sys.argv) == 3 and sys.argv[1] == 'rescore':
        rescore(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == 'compare':
        sys.exit(1 if compare(sys.argv[2], sys.argv[3]) else 0)
    else:
        raise SystemExit(__doc__)
