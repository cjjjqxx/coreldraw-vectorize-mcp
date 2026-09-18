"""CorelDRAW automation MCP server (APIs verified against CorelDRAW 2022 v24).

Drives a locally running CorelDRAW instance through its COM automation
interface. IMPORTANT: connect to a VISIBLE, already-initialized CorelDRAW
window. This server NEVER auto-launches CorelDRAW: /Automation headless
startup is unreliable and repeated attempts spawn runaway instances, so the
caller must open CorelDRAW 2022 (a visible window) first and then call a tool.

Verified API facts (probed from the installed typelib):
  - document creation: app.CreateDocumentEx(app.CreateStructCreateOptions())
  - shapes: layer.CreateRectangle2(x, y, width, height)
            layer.CreateEllipse2(cx, cy, rx, ry)
            layer.CreateLineSegment(x1, y1, x2, y2)
            layer.CreateArtisticText(x, y, text, langId, charSet, font, size,
                                     bold, italic, underline, align)
  - fill:    col = app.CreateColor(); col.RGBAssign(r,g,b)
             shape.Fill.ApplyUniformFill(col)
  - export:  doc.Export(file, 802 /*cdrPNG*/, 1 /*cdrCurrentPage*/, None, None)

  NOTE: cdrExportRange is 0=cdrAllPages, 1=cdrCurrentPage, 2=cdrSelection.
  Passing 2 exports only the current SELECTION, which silently crops output.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

# The tracing/checking helpers live next to this file; make them importable
# regardless of the working directory the host spawns us in.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("coreldraw")

# Only used by cdr_launch. Set CDR_EXE if CorelDRAW is installed elsewhere.
CDR_EXE = os.environ.get(
    "CDR_EXE",
    r"C:\Program Files\Corel\CorelDRAW Graphics Suite 2022\Programs64\CorelDRW.exe",
)
# Default output folder for runs without an explicit work_dir. Override with CDR_WORK.
WORK_DIR = Path(os.environ.get("CDR_WORK", str(Path.home() / "cdr-vectorize-work")))

# cdrFilter / cdrExportRange enum values, read from the installed typelib.
CDR_PNG = 802
CDR_RANGE_PAGE = 1   # cdrCurrentPage -- the whole page (NOT 2 = cdrSelection!)
CDR_RANGE_ALL = 0    # cdrAllPages


def _ensure_workdir() -> Path:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    return WORK_DIR


def _ensure_com() -> None:
    """Ensure COM is initialized on this thread. The MCP server runs tool
    handlers on its own thread; without an explicit CoInitialize the COM
    Dispatch calls fail with '尚未调用 CoInitialize' (not initialized)."""
    import pythoncom

    try:
        pythoncom.CoInitialize()
    except Exception:  # noqa: BLE001 - already initialized is fine
        pass


def _is_cdr_ready() -> bool:
    """True only if a CorelDRAW instance is already running and responds.
    Never launches anything."""
    import win32com.client

    _ensure_com()
    try:
        app = win32com.client.Dispatch("CorelDRAW.Application")
        _ = app.Documents.Count
        return True
    except Exception:  # noqa: BLE001
        return False


def _launch_once() -> bool:
    """Launch a single visible CorelDRAW instance if none is running.
    Returns True if it was launched by this call, False if one was already
    running (or launch failed). This NEVER loops and is guarded so a repeated
    call never starts a second instance."""
    import subprocess

    if _is_cdr_ready():
        return False
    try:
        subprocess.Popen(
            [CDR_EXE], cwd=str(Path(CDR_EXE).parent),
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        return True
    except Exception:  # noqa: BLE001
        return False


@mcp.tool()
def cdr_launch() -> dict[str, Any]:
    """Launch a single visible CorelDRAW 2022 instance (automation-ready) if
    none is already running. Polls until the COM interface answers. Safe to
    call repeatedly: it never spawns more than one instance. Returns whether
    it launched a new instance or attached to an existing one."""
    import win32com.client

    _ensure_com()
    launched = _launch_once()
    attempts = 0
    last = None
    while attempts < 12:  # ~24s max for first-run init
        try:
            app = win32com.client.Dispatch("CorelDRAW.Application")
            _ = app.Documents.Count
            return {
                "ok": True,
                "launched_new": launched,
                "name": str(app.Name),
                "version": str(app.Version),
                "document_count": int(app.Documents.Count),
            }
        except Exception as exc:  # noqa: BLE001
            last = exc
            attempts += 1
            time.sleep(2)
    raise RuntimeError(f"CorelDRAW did not become ready: {last}")


def _connect() -> Any:
    """Connect to an ALREADY-RUNNING visible CorelDRAW instance.

    Important: this never launches a new CorelDRAW process. /Automation
    headless startups are unreliable and, worse, repeated attempts spawn
    runaway instances. If no instance is running, we raise immediately with a
    clear message telling the caller to open CorelDRAW first.
    """
    import win32com.client

    _ensure_com()
    attempts = 0
    last = None
    # Poll for a short window: an instance may still be finishing its
    # first-run initialization, but we never trigger a launch ourselves.
    while attempts < 3:
        try:
            app = win32com.client.Dispatch("CorelDRAW.Application")
            # require an initialized Documents collection to consider healthy
            _ = app.Documents.Count
            return app
        except Exception as exc:  # noqa: BLE001
            last = exc
            attempts += 1
            time.sleep(2)
    raise RuntimeError(
        "CorelDRAW COM connect failed. Open CorelDRAW 2022 first (a visible "
        f"window), then retry. Last error: {last}"
    )


def _get_doc(app: Any) -> Any:
    if int(app.Documents.Count) == 0:
        opts = app.CreateStructCreateOptions()
        app.CreateDocumentEx(opts)
    return app.ActiveDocument


def _fill(shape: Any, app: Any, hex_color: str) -> bool:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    try:
        col = app.CreateColor()
        col.RGBAssign(r, g, b)
        shape.Fill.ApplyUniformFill(col)
        return True
    except Exception:  # noqa: BLE001
        return False


@mcp.tool()
def cdr_status() -> dict[str, Any]:
    """Check CorelDRAW automation availability and report version plus the
    number of open documents."""
    try:
        app = _connect()
        return {
            "ok": True,
            "name": str(app.Name),
            "version": str(app.Version),
            "document_count": int(app.Documents.Count),
            "exe": CDR_EXE,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def _auto_export(app: Any, document_index: int, out_path: str = "") -> dict[str, Any]:
    """Export the given document to PNG and save it as a .cdr, autorecovering
    sensible filenames. Shared by the drawing tools so they persist results.
    Exporting an EMPTY document fails in CorelDRAW, so this reports that
    instead of letting the whole tool call crash."""
    work = _ensure_workdir()
    stamp = int(time.time())
    png = out_path or str(work / f"cdr_autosave_{stamp}.png")
    doc = app.Documents.Item(document_index + 1)
    try:
        doc.Export(png, CDR_PNG, CDR_RANGE_PAGE, None, None)
    except Exception as exc:  # noqa: BLE001
        # empty doc (no shapes) -> Export raises "发生意外"
        return {
            "ok": False,
            "error": f"export skipped/blocked: {exc}",
            "hint": "add a shape first or use an existing document with content",
        }
    cdr = str(work / f"cdr_autosave_{stamp}.cdr")
    try:
        opts = app.CreateStructSaveAsOptions()
        doc.SaveAs(cdr, opts)
    except Exception:  # noqa: BLE001
        try:
            doc.SaveAs(cdr)
        except Exception:  # noqa: BLE001
            cdr = ""
    p = Path(png)
    return {
        "ok": True,
        "png": str(p),
        "png_bytes": p.stat().st_size if p.exists() else 0,
        "cdr": cdr,
    }


@mcp.tool()
def cdr_new_document(
    auto_export: bool = True, out_path: str = "",
) -> dict[str, Any]:
    """Create a new empty CorelDRAW document. Returns document name and the
    new total count. Note: this does NOT auto-export (an empty document cannot
    be exported by CorelDRAW); export happens once shapes are drawn."""
    app = _connect()
    opts = app.CreateStructCreateOptions()
    app.CreateDocumentEx(opts)
    doc = app.ActiveDocument
    idx = int(app.Documents.Count) - 1
    return {
        "ok": True,
        "document": str(doc.Name),
        "pages": int(app.Documents.Count),
        "document_index": idx,
    }


@mcp.tool()
def cdr_draw_rectangle(
    x: float, y: float, width: float, height: float, fill_hex: str = "FF0000",
    auto_export: bool = True, out_path: str = "",
) -> dict[str, Any]:
    """Draw a filled rectangle from top-left (x, y) with the given width and
    height on the active page. fill_hex is RRGGBB (e.g. 3366CC). Units are the
    document's default page units. If auto_export (default True) the document
    is saved as .cdr and exported to PNG automatically."""
    app = _connect()
    doc = _get_doc(app)
    layer = doc.ActiveLayer
    shape = layer.CreateRectangle2(x, y, width, height)
    filled = _fill(shape, app, fill_hex)
    idx = int(app.Documents.Count) - 1
    result = {
        "ok": True,
        "type": "rectangle",
        "x": x, "y": y, "width": width, "height": height,
        "fill": fill_hex,
        "filled": filled,
        "document_index": idx,
    }
    if auto_export:
        result["export"] = _auto_export(app, idx, out_path)
    return result


@mcp.tool()
def cdr_draw_ellipse(
    cx: float, cy: float, rx: float, ry: float, fill_hex: str = "0000FF",
    auto_export: bool = True, out_path: str = "",
) -> dict[str, Any]:
    """Draw a filled ellipse centered at (cx, cy) with radii rx and ry.
    fill_hex is RRGGBB. If auto_export (default True) the document is saved as
    .cdr and exported to PNG automatically."""
    app = _connect()
    doc = _get_doc(app)
    layer = doc.ActiveLayer
    shape = layer.CreateEllipse2(cx, cy, rx, ry)
    filled = _fill(shape, app, fill_hex)
    idx = int(app.Documents.Count) - 1
    result = {
        "ok": True,
        "type": "ellipse",
        "cx": cx, "cy": cy, "rx": rx, "ry": ry,
        "fill": fill_hex,
        "filled": filled,
        "document_index": idx,
    }
    if auto_export:
        result["export"] = _auto_export(app, idx, out_path)
    return result


@mcp.tool()
def cdr_draw_line(
    x1: float, y1: float, x2: float, y2: float,
    auto_export: bool = False, out_path: str = "",
) -> dict[str, Any]:
    """Draw a straight line segment from (x1,y1) to (x2,y2) on the active
    page. Use this for diagonals such as rig legs and cables."""
    app = _connect()
    doc = _get_doc(app)
    layer = doc.ActiveLayer
    layer.CreateLineSegment(x1, y1, x2, y2)
    idx = int(app.Documents.Count) - 1
    result = {
        "ok": True, "type": "line",
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "document_index": idx,
    }
    if auto_export:
        result["export"] = _auto_export(app, idx, out_path)
    return result


@mcp.tool()
def cdr_draw_polyline(
    points: list[float], close: bool = False,
    auto_export: bool = False, out_path: str = "",
) -> dict[str, Any]:
    """Draw a polyline through a flat list of [x0,y0,x1,y1,...] coordinates by
    chaining straight line segments. If close is True the last point is joined
    back to the first."""
    app = _connect()
    doc = _get_doc(app)
    layer = doc.ActiveLayer
    if len(points) < 4 or len(points) % 2 != 0:
        return {"ok": False, "error": "points must be a flat even-length coordinate list"}
    coords = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
    seq = coords + [coords[0]] if close else coords
    segments = 0
    for i in range(len(seq) - 1):
        (ax, ay), (bx, by) = seq[i], seq[i + 1]
        layer.CreateLineSegment(ax, ay, bx, by)
        segments += 1
    idx = int(app.Documents.Count) - 1
    result = {
        "ok": True,
        "vertices": len(coords),
        "segments": segments,
        "closed": close,
        "document_index": idx,
    }
    if auto_export:
        result["export"] = _auto_export(app, idx, out_path)
    return result


@mcp.tool()
def cdr_add_text(
    x: float, y: float, text: str, font_size: float = 10.0,
    bold: bool = False, auto_export: bool = False, out_path: str = "",
) -> dict[str, Any]:
    """Place an artistic text label at (x, y) on the active page. Supports CJK
    text so diagram labels can be written in Chinese."""
    app = _connect()
    doc = _get_doc(app)
    layer = doc.ActiveLayer
    # CreateArtisticText(x, y, Text, LanguageID, CharSet, FontName,
    #                    FontSize, Bold, Italic, Underline, Alignment)
    shape = layer.CreateArtisticText(
        x, y, text, 0, 0, "", font_size, bold, False, False, 0,
    )
    idx = int(app.Documents.Count) - 1
    result = {
        "ok": True, "type": "text", "text": text,
        "x": x, "y": y, "font_size": font_size,
        "document_index": idx,
    }
    if auto_export:
        result["export"] = _auto_export(app, idx, out_path)
    return result


@mcp.tool()
def cdr_import_image(image_path: str) -> dict[str, Any]:
    """Import a raster image (PNG/JPG) onto the active page as a reference
    shape for manual tracing. Returns the new shape count."""
    app = _connect()
    doc = _get_doc(app)
    layer = doc.ActiveLayer
    p = Path(image_path)
    if not p.exists():
        return {"ok": False, "error": f"image not found: {image_path}"}
    shape = layer.CreateImportFromFile(str(p))
    return {"ok": True, "shape_id": int(shape.ShapeID), "file": str(p)}


@mcp.tool()
def cdr_export_png(document_index: int = 0, out_path: str = "", dpi: int = 96) -> dict[str, Any]:
    """Export the document to a PNG file. Returns path and byte size.
    document_index is 0-based; out_path defaults to the work dir."""
    app = _connect()
    if int(app.Documents.Count) == 0:
        return {"ok": False, "error": "no open document"}
    work = _ensure_workdir()
    target = out_path or str(work / f"cdr_export_{int(time.time())}.png")
    doc = app.Documents.Item(document_index + 1)
    # Export(FileName, Filter=cdrPNG, Range=all, Options=None, PaletteOptions=None)
    doc.Export(target, CDR_PNG, CDR_RANGE_PAGE, None, None)
    p = Path(target)
    return {
        "ok": True,
        "png": str(p),
        "exists": p.exists(),
        "bytes": p.stat().st_size if p.exists() else 0,
    }


@mcp.tool()
def cdr_save_document(out_path: str = "", document_index: int = 0) -> dict[str, Any]:
    """Save the document as a .cdr file (default: work dir with a timestamp)."""
    app = _connect()
    if int(app.Documents.Count) == 0:
        return {"ok": False, "error": "no open document"}
    work = _ensure_workdir()
    target = out_path or str(work / f"cdr_doc_{int(time.time())}.cdr")
    doc = app.Documents.Item(document_index + 1)
    doc.SaveAs(target, app.CreateStructSaveAsOptions())  # 1-arg form fails: options struct is required
    p = Path(target)
    return {"ok": True, "cdr": str(p), "exists": p.exists()}


@mcp.tool()
def cdr_trace_image(
    image_path: str,
    mode: str = "smooth",
    thresh: int = 175,
    detail: float = 0.5,
    out_stem: str = "",
) -> dict[str, Any]:
    """Vectorize a raster drawing into clean, EDITABLE vector artwork and open
    it in CorelDRAW, then export PNG + CDR.

    mode="smooth" (default, best for technical drawings with text):
        filled contours with the pixel staircase removed by low-pass filtering.
        Glyph-bearing shapes are deliberately NOT smoothed (smoothing merges
        adjacent strokes and makes CJK labels illegible) while large shapes get
        two smoothing passes. Text stays readable and linework looks crisp.
    mode="stroke":
        skeleton centerlines drawn as uniform-width strokes. Sharpest possible
        lines and fewest nodes, but it reduces glyphs to single-line skeletons
        and turns solid areas into outlines - do not use it for labelled drawings.
    mode="outline":
        raw contour tracing with no smoothing. Exact but keeps every
        anti-aliased pixel edge, so it looks as fuzzy as the source.

    Verify the result with cdr_compare / cdr_reproduce.
    """
    p = Path(image_path)
    if not p.exists():
        return {"ok": False, "error": f"image not found: {image_path}"}
    work = _ensure_workdir()
    stem = out_stem or p.stem
    svg = work / f"{stem}_traced.svg"
    png = work / f"{stem}_traced.png"
    cdr = work / f"{stem}_traced.cdr"

    if mode == "outline":
        import cdr_trace
        info = cdr_trace.vectorize(str(p), str(svg), simplify=1.2, thresh=thresh)
    elif mode == "stroke":
        import cdr_trace_skeleton
        info = cdr_trace_skeleton.vectorize_strokes(
            str(p), str(svg), thresh=thresh, simplify=1.0, smooth=True, min_len=4)
    else:
        import cdr_trace_smooth
        info = cdr_trace_smooth.vectorize_smooth(
            str(p), str(svg), thresh=thresh, detail=detail,
            smooth_small=0, smooth_large=2)

    app = _connect()
    doc = app.OpenDocument(str(svg))
    doc.Export(str(png), CDR_PNG, CDR_RANGE_PAGE, None, None)
    saved = False
    try:
        doc.SaveAs(str(cdr), app.CreateStructSaveAsOptions())
        saved = True
    except Exception:  # noqa: BLE001
        pass
    return {
        "ok": True,
        "method": f"vector-{mode}",
        "svg": str(svg),
        "png": str(png),
        "png_bytes": png.stat().st_size if png.exists() else 0,
        "cdr": str(cdr) if saved else "",
        "outer_contours": info.get("outer_contours", info.get("paths")),
        "hole_contours": info.get("hole_contours"),
        "points": info.get("points", info.get("points_total")),
        "source_size": info.get("size"),
    }


@mcp.tool()
def cdr_compare(
    reference_path: str, candidate_path: str,
    tolerance: int = 4, grid_cols: int = 12, grid_rows: int = 9,
) -> dict[str, Any]:
    """CHECK step: quantify how close a reproduction is to the reference.
    Returns recall (how much of the reference linework is covered), precision
    (how much drawn linework is actually valid) and F1, plus a per-cell map of
    the worst regions so the next correction targets real gaps."""
    import cdr_check

    r = cdr_check.compare(
        reference_path, candidate_path,
        tolerance=tolerance, grid=(grid_cols, grid_rows),
    )
    return {
        "ok": True,
        "recall": r["recall"],
        "precision": r["precision"],
        "f1": r["f1"],
        "reference_px": r["reference_px"],
        "candidate_px": r["candidate_px"],
        "grid_map": cdr_check.ascii_report(r),
    }


@mcp.tool()
def cdr_reproduce(
    image_path: str,
    tolerance: int = 4,
    mode: str = "smooth",
    detail: float = 0.5,
    min_f1: float = 0.9,
) -> dict[str, Any]:
    """End-to-end REPRODUCE + CHECK: vectorize the reference image into
    CorelDRAW, export it, then immediately score the export against the
    reference and report whether it meets min_f1. The metric breakdown is
    always returned, so a miss is visible instead of being declared done."""
    traced = cdr_trace_image(image_path, mode=mode, detail=detail)
    if not traced.get("ok"):
        return traced
    verdict = cdr_compare(image_path, traced["png"], tolerance=tolerance)
    verdict.update({
        "png": traced["png"],
        "cdr": traced["cdr"],
        "svg": traced["svg"],
        "outer_contours": traced.get("outer_contours"),
        "hole_contours": traced.get("hole_contours"),
        "passed": bool(verdict.get("f1", 0) >= min_f1),
        "min_f1": min_f1,
    })
    return verdict


def _run_step(args: list[str], timeout: int) -> tuple[int, str, str]:
    """Run a vectorize step in a child python: COM/SR/OCR hangs must not freeze this server."""
    import subprocess

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [sys.executable, *args], cwd=str(Path(__file__).resolve().parent),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout, env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return proc.returncode, proc.stdout, proc.stderr


def _review_payload(meta: dict, work: Path) -> dict[str, Any]:
    """What the calling AI must look at: uncertain/weak labels (L#) and unrecognized text (M#)."""
    check = []
    for i, c in enumerate(meta.get("candidates", [])):
        weak = not c.get("use", True) and c.get("skip_reason", "").startswith("single weak")
        if (c.get("use", True) and c.get("uncertain")) or weak:
            alts = []
            for r in c.get("readings", []):
                if r[0] not in alts and r[0] != c["text"]:
                    alts.append(r[0])
            check.append({"id": f"L{i}", "text": c["text"], "box": c["box"], "placed": bool(c.get("use", True)),
                          "why": c.get("why") or c.get("skip_reason", ""), "other_readings": alts[:3]})
    unlabeled = [{"id": f"M{j}", "box": m["box"]} for j, m in enumerate(meta.get("unlabeled_text", []))]
    return {"review_sheet": meta.get("review_sheet"), "labels_to_check": check, "unlabeled_text": unlabeled}


_FIX_HINT = ("Open review_sheet (one image: source crop + current reading per item). For each L# that is "
             "wrong fix its text (or drop it); add not-placed L# you want (text + box); for each M# that is "
             "real text add {text, box}. Then call cdr_vectorize_build(work_dir, labels=<corrected list>) - "
             "start from `labels` returned here. Items that are fine need no action.")


@mcp.tool()
def cdr_vectorize_prepare(image_path: str, work_dir: str = "") -> dict[str, Any]:
    """STEP 1 of image -> editable CDR (line drawings, schematics, colour maps with labels; blurry,
    small or JPEG input is fine).

    Runs AI super-resolution for the graphics and multi-scale OCR for the labels, then decides on
    its own which OCR results become text objects. Returns:
      labels            ready-to-use list for cdr_vectorize_build (no review needed to proceed)
      mode              'color' or 'lineart', detected from the image
      labels_to_check   uncertain readings (L#), with the reason and alternative readings
      unlabeled_text    glyph-like ink no label covers (M#) - possibly text OCR missed
      review_sheet      ONE image showing every L#/M# crop with its current reading
    Boxes are [x0, y0, x1, y1] in pixels of source_png (large inputs are shrunk first: source_scale).
    Prefer cdr_vectorize for the one-call flow."""
    import json

    # absolute paths only: CorelDRAW resolves relative paths against its own working directory and
    # fails with a bare "发生意外" (unexpected error) on Import
    p = Path(image_path).resolve()
    if not p.exists():
        return {"ok": False, "error": f"image not found: {image_path}"}
    work = Path(work_dir).resolve() if work_dir else _ensure_workdir() / f"vec_{p.stem}_{int(time.time())}"
    work.mkdir(parents=True, exist_ok=True)
    try:
        code, out, err = _run_step(["cdr_vectorize.py", "prepare", str(p), str(work)], timeout=900)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"prepare step failed/timed out: {exc}", "work_dir": str(work)}
    meta_path = work / "ocr_candidates.json"
    if code != 0 or not meta_path.exists():
        return {"ok": False, "error": (err or out)[-1500:], "work_dir": str(work)}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    import cdr_vectorize
    return {
        "ok": True,
        "work_dir": str(work),
        "source_size": meta["src_size"],
        "source_png": str(work / "src.png"),
        # boxes refer to source_png; a large input is shrunk first (source_scale < 1 = input px * scale)
        "source_scale": meta.get("source_scale", 1.0),
        "mode": meta.get("mode", "lineart"),
        "labels": cdr_vectorize.suggested_labels(meta),
        **_review_payload(meta, work),
        "next": "cdr_vectorize_build(work_dir) uses `labels` as-is. " + _FIX_HINT,
    }


@mcp.tool()
def cdr_vectorize_build(
    work_dir: str,
    labels: list[dict[str, Any]] | None = None,
    font: str = "新宋体",
    trace_type: str = "lineart",
    detail: int = 100,
    smoothing: int = 25,
    mode: str = "auto",
    colors: int = 12,
) -> dict[str, Any]:
    """STEP 2 of image -> editable CDR. Needs a running CorelDRAW 2022 and a work_dir from
    cdr_vectorize_prepare (or cdr_vectorize).

    labels: omit to use the labels prepare selected; or pass a corrected list
    [{"text": "...", "box": [x0, y0, x1, y1], "angle": deg (optional)}] in source pixels ("a|b" splits
    one box into several labels; [] = no text). Labels are wiped from the image (crossing lines kept),
    PowerTRACE vectorizes the graphics and each label becomes an editable text object sized, rotated
    and coloured like the original.
    mode: 'auto' (default, from the image), 'lineart' (black linework + text) or 'color' (flat colour
    layers + grey graticule + black ink + text). `colors` = palette size for colour mode.
    font: '新宋体' / '仿宋' / '黑体' / '微软雅黑'. Returns result.cdr / result.png plus a graphic self-check:
    `issues` (trace_failed / missing_graphics / extra_graphics / colour_mismatch, with regions in
    source_png px and a hint) and self_check.diff_png (red = missing, blue = extra)."""
    import json

    work = Path(work_dir).resolve()          # relative paths break CorelDRAW's Import (see prepare)
    if not (work / "sr2.png").exists():
        return {"ok": False, "error": f"not a prepared work_dir (run cdr_vectorize_prepare first): {work_dir}"}
    meta_prep = {}
    if (work / "ocr_candidates.json").exists():
        meta_prep = json.loads((work / "ocr_candidates.json").read_text(encoding="utf-8"))
    if labels is None:
        import cdr_vectorize
        labels = cdr_vectorize.suggested_labels(meta_prep)
    if mode == "auto":
        mode = meta_prep.get("mode", "lineart")
    clean = []
    for lab in labels or []:
        box = lab.get("box")
        if not lab.get("text") or not isinstance(box, (list, tuple)) or len(box) != 4:
            return {"ok": False, "error": f"each label needs text and box [x0,y0,x1,y1]: {lab}"}
        item = {"text": str(lab["text"]), "box": [int(v) for v in box]}
        if "angle" in lab:
            item["angle"] = float(lab["angle"])
        clean.append(item)
    confirmed = work / "labels_confirmed.json"
    confirmed.write_text(json.dumps(clean, ensure_ascii=False, indent=1), encoding="utf-8")
    is_color = mode == "color"
    prep_args = (["cdr_vectorize_color.py", "layers", str(work), str(confirmed), str(colors)]
                 if is_color else ["cdr_vectorize.py", "finalize", str(work), str(confirmed)])
    meta_file = "layers.json" if is_color else "labels.json"
    try:
        code, out, err = _run_step(prep_args, timeout=280)
        if code != 0 or not (work / meta_file).exists():
            return {"ok": False, "step": "prepare-layers" if is_color else "finalize",
                    "error": (err or out)[-1500:]}
        if not _is_cdr_ready():
            return {"ok": False, "step": "coreldraw",
                    "error": "CorelDRAW is not running/responding. Open CorelDRAW 2022, then call this again."}
        com_args = ["cdr_vectorize_com.py", str(work), font, trace_type, str(detail), str(smoothing)]
        if is_color:
            com_args.append("color")
        for attempt in range(2):
            import cdr_vectorize_com
            code, out, err = _run_step(com_args, timeout=cdr_vectorize_com.time_budget(str(work), mode) + 60)
            lines = [ln for ln in out.strip().splitlines() if ln.startswith("{")]
            result = json.loads(lines[-1]) if lines else {"ok": False, "error": (err or out)[-1500:]}
            # CorelDRAW intermittently rejects a COM call while busy ("发生意外"); the identical call
            # succeeds a few seconds later, so retry once before reporting a failure
            if result.get("ok") or "com_error" not in str(result.get("error", "")) or attempt:
                break
            time.sleep(5)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"build step failed/timed out: {exc}"}
    if not result.get("ok"):
        result.setdefault("step", "coreldraw")
        return result   # includes "trace" (last lines of the CorelDRAW-side traceback)
    trace_stats = result.pop("layers_built", None)
    try:
        import cdr_selfcheck
        chk = cdr_selfcheck.self_check(str(work), trace_stats)
        result["issues"] = chk["issues"]            # graphic problems for the caller to act on ([] = none)
        result["self_check"] = {**chk["metrics"], "diff_png": chk["diff_png"]}
    except Exception as exc:  # noqa: BLE001 - the check must never fail a finished build
        result["issues"] = [{"type": "self_check_error", "detail": f"{type(exc).__name__}: {exc}"}]
    result["mode"] = mode
    result["labels_placed"] = len(clean)
    result["work_dir"] = str(work)
    return result


@mcp.tool()
def cdr_vectorize(image_path: str, work_dir: str = "", mode: str = "auto", font: str = "新宋体") -> dict[str, Any]:
    """ONE CALL image -> editable CorelDRAW file (line drawings, schematics, colour maps with text;
    blurry / small / JPEG input is fine). Needs a running CorelDRAW 2022.

    Super-resolves and traces the graphics, OCRs the labels and re-creates them as editable text,
    picks colour vs line-art mode itself, saves result.cdr + result.png. It also reports what it is
    NOT sure about so you can fix it without redoing anything:
      labels_to_check / unlabeled_text / review_sheet  (see cdr_vectorize_prepare)
    needs_review=false: nothing uncertain. Otherwise look at review_sheet and, if anything is wrong,
    call cdr_vectorize_build(work_dir, labels=<corrected `labels`>) - super-resolution and OCR are
    reused, only layers/trace/text are rebuilt."""
    prep = cdr_vectorize_prepare(image_path, work_dir)
    if not prep.get("ok"):
        return prep
    build = cdr_vectorize_build(prep["work_dir"], None, font=font, mode=mode)
    if not build.get("ok"):
        build["work_dir"] = prep["work_dir"]
        return build
    needs_review = bool(prep["labels_to_check"] or prep["unlabeled_text"] or build.get("issues"))
    return {
        **build,
        "labels": prep["labels"],
        "labels_to_check": prep["labels_to_check"],
        "unlabeled_text": prep["unlabeled_text"],
        "review_sheet": prep["review_sheet"],
        "needs_review": needs_review,
        "next": ((("Graphic issues: see `issues` and self_check.diff_png. " if build.get("issues") else "")
                  + (_FIX_HINT if (prep["labels_to_check"] or prep["unlabeled_text"]) else ""))
                 or "No uncertain items.") + " Always look at result.png before reporting success.",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
