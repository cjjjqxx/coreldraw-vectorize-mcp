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

from mcp.server.mcpserver import Image, MCPServer
from mcp_types import TextContent

# Server-level guidance. Clients that honour MCP `instructions` put this in front of the model when
# they connect, so the workflow stops depending on the caller having read the repository README - which
# an MCP client never shows it. Not every client forwards `instructions`, which is why the same
# guidance is repeated in the tool results and the pictures travel with them.
WORKFLOW_INSTRUCTIONS = """\
cdr turns a raster drawing into an editable CorelDRAW file. ONE call is only the FIRST PASS: the call
reports what it is unsure about, and the result is markedly worse if those reports are ignored.

1. Call cdr_vectorize(image_path). Read `workflow` FIRST - it is the first field of every answer and it
   says whether the job is finished, half-done, or still running. The call returns result.cdr /
   result.png and, as real pictures in this conversation, review_sheet (every uncertain label with its
   current reading) and diff_png (red = in the source but missing from the result, blue = drawn but not
   in the source). If it returns done=false with status="running", nothing failed: the work is still
   going, so call cdr_job_status(work_dir) again in 30-60 s.
2. While workflow.step is "review":
   a. Look at review_sheet. For every L# correct the reading (or drop it if it is not text); for every
      M# add the missed label. Put "a|b|c" in one box to split it into several labels.
   b. Treat graphics_issues and text_issues separately: a tracing fault is fixed with mode/colors, a
      label fault with text/box. Fixing one never fixes the other.
   c. Fix labels with label_patch - do NOT re-send the whole list, which is where entries get lost:
      cdr_vectorize_build(work_dir, label_patch={"set": {"L3": "藏"}, "drop": ["L7"], "add": [...]})
   d. Prefer editing `text` over `box`. A label box also decides which pixels are wiped from the image
      before it is traced, so a wrong box destroys linework no later step can restore (wipe_spill
      reports exactly that).
   e. Repeat a-d until workflow.step is "converged", and at most 3 rounds.
   f. Leave `to_fill` alone. Those spots are DELIBERATELY blank - an unreadable reading, or a slanted
      label whose text length could not be measured (guessing its size from the box would print it at
      several times the right size). The CDR marks each with a magenta box on the "TO FILL" layer, and
      `to_fill_marks` counts them. They are not failures, so re-running does not fill them: type them
      in CorelDRAW and delete that layer. Say so in your answer instead of retrying.
3. Call cdr_finish(work_dir). It REFUSES to confirm delivery while anything above is unhandled, so a
   half-reviewed conversion cannot be reported to the user as finished. When it does deliver, put
   `remaining_issues` into your answer honestly and say which parts are better finished by hand in
   CorelDRAW (font, size, spaces lost by OCR)."""

mcp = MCPServer("coreldraw", instructions=WORKFLOW_INSTRUCTIONS)

# Low-level CorelDRAW primitives are NOT registered by default. With twenty tools side by side, a
# caller that cannot tell the main path from the primitives reaches for cdr_trace_image (which traces
# shapes but cannot rebuild text) or starts hand-drawing with cdr_draw_* - a far worse result with no
# sign that anything went wrong. Set CDR_EXPOSE_PRIMITIVES=1 to get them back for scripting.
EXPOSE_PRIMITIVES = os.environ.get("CDR_EXPOSE_PRIMITIVES", "").strip().lower() in ("1", "true", "yes")


def _primitive(fn: Any) -> Any:
    """Register a low-level tool only when CDR_EXPOSE_PRIMITIVES is on."""
    return mcp.tool()(fn) if EXPOSE_PRIMITIVES else fn

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


AUTO_RESTART = os.environ.get("CDR_NO_AUTO_RESTART", "").strip() in ("", "0", "false", "False")
# 2500 s was too late: a grid map whose COM step takes 84 s in a fresh instance ran past its
# budget in one that had burnt roughly 1500-1700 s, and the build was lost. A restart costs ~35 s.
CPU_RESTART_S = float(os.environ.get("CDR_CPU_RESTART_S", "1200") or 1200)


def _cdr_ping(timeout: float = 20.0) -> bool:
    """Is the running instance answering? Asked in a CHILD process, with a timeout.

    A hung CorelDRAW does not refuse COM calls - it never answers them, and `app.Documents.Count`
    then blocks forever. Asking in-process (which is what the old check did) therefore wedged the
    server on exactly the instance it was supposed to detect. A child can simply be killed.
    """
    import subprocess
    import sys as _sys

    # Documents.Count answers before the application is actually usable: a freshly started instance
    # then fails the build on `app.FontList` ("does not support enumeration"). The font list is the
    # last thing to come up, so it is what readiness is measured by.
    code = ("import pythoncom, win32com.client;pythoncom.CoInitialize();"
            "a=win32com.client.Dispatch('CorelDRAW.Application');"
            "n=int(a.Documents.Count);f=len([str(x) for x in a.FontList]);"
            "m=[getattr(a, x) for x in ('CreateDocumentEx','CreateStructCreateOptions')];"
            "print(n, f, len(m));raise SystemExit(0 if f and len(m) == 2 else 3)")
    try:
        r = subprocess.run([_sys.executable, "-c", code], capture_output=True, timeout=timeout)
        return r.returncode == 0
    except Exception:  # noqa: BLE001 - timeout, or no instance at all
        return False


def _cdr_cpu_seconds() -> float:
    """CPU time the CorelDRAW process has burnt, or 0. It climbs with every trace; past roughly an
    hour of CPU the instance starts refusing to finish COM calls (measured over this project's runs),
    so a build can restart it BEFORE it hangs rather than after."""
    import subprocess

    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-Process CorelDRW -ErrorAction SilentlyContinue | "
             "Measure-Object -Property CPU -Sum).Sum"],
            capture_output=True, text=True, timeout=20)
        return float((out.stdout or "0").strip() or 0)
    except Exception:  # noqa: BLE001
        return 0.0


def _kill_cdr() -> None:
    import subprocess

    try:
        subprocess.run(["taskkill", "/F", "/IM", "CorelDRW.exe"], capture_output=True, timeout=30)
    except Exception:  # noqa: BLE001
        pass
    time.sleep(3)


def ensure_cdr_healthy(reason_out: dict[str, Any] | None = None) -> str:
    """Make sure the build has a CorelDRAW that will answer. Returns what was done.

    Two failure modes, both seen in this project's runs: the instance is already hung (COM calls
    never return), and the instance is still answering but has burnt so much CPU that it will hang
    part-way through the next build - which costs the whole build, not just a restart.
    """
    if not AUTO_RESTART:
        return "disabled"
    cpu = _cdr_cpu_seconds()
    tired = cpu > CPU_RESTART_S
    alive = _cdr_ping()
    if alive and not tired:
        return "ok"
    if alive and tired:
        try:                                   # a tired instance still answers: let it close cleanly
            import win32com.client
            _ensure_com()
            app = win32com.client.Dispatch("CorelDRAW.Application")
            for _ in range(int(app.Documents.Count)):
                app.ActiveDocument.Close()
        except Exception:  # noqa: BLE001
            pass
    _kill_cdr()
    _launch_once()
    for _ in range(20):                      # a cold start needs ~20-40 s before the fonts are up
        if _cdr_ping(timeout=15):
            break
        time.sleep(3)
    what = f"restarted ({'cpu %.0fs' % cpu if tired else 'unresponsive'})"
    if isinstance(reason_out, dict):
        reason_out["cdr_restarted"] = what
    return what


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


@_primitive
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


@_primitive
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


@_primitive
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


@_primitive
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


@_primitive
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


@_primitive
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


@_primitive
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


@_primitive
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


@_primitive
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


@_primitive
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


@_primitive
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


@_primitive
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


# ---------------------------------------------------------------------------
# Long steps run as background children behind a bounded wait.
#
# The old code let a step run for up to 900 s while MCP clients give up after 60 s by default. The
# caller then saw "-32001 Request timed out" and got no work_dir back, while the server kept burning
# CPU on a result nobody could reach - a failure the caller could neither diagnose nor resume. Now
# every step is a child process whose state lives in the work_dir, the synchronous wait is budgeted,
# and running out of budget returns a resumable handle instead of silence.
# ---------------------------------------------------------------------------

JOB_FILE = "_job.json"
DEFAULT_WAIT_SECONDS = 45.0     # deliberately below the common 60 s MCP client timeout
MAX_BUILD_ROUNDS = 3            # the review loop may rebuild, but not forever
UNCERTAIN_HINT_MIN = 12         # from this many uncertain readings on, offer the blank-out option
MAX_LIST_ITEMS = 40             # per-list cap in an answer: a long JSON is what makes a client drop
                                # the inline pictures, which are the whole point of carrying them


def _job_write(work: Path, **fields: Any) -> None:
    """Record job state in the work_dir so a handed-off call can be picked up later."""
    import json

    p = work / JOB_FILE
    cur: dict[str, Any] = {}
    try:
        if p.exists():
            cur = json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a torn file is not worth failing over
        cur = {}
    cur.update(fields)
    cur["updated"] = time.time()
    try:
        p.write_text(json.dumps(cur, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def _job_read(work: Path) -> dict[str, Any]:
    import json

    try:
        p = work / JOB_FILE
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:  # noqa: BLE001
        return {}


def _pid_alive(pid: int) -> bool:
    """Whether a child we started is still running. Windows API, so there is no psutil dependency."""
    if not pid:
        return False
    try:
        import ctypes

        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, int(pid))          # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            ok = bool(k.GetExitCodeProcess(h, ctypes.byref(code)))
            return ok and code.value == 259                 # STILL_ACTIVE
        finally:
            k.CloseHandle(h)
    except Exception:  # noqa: BLE001 - an unanswerable probe means "gone"
        return False


def _start_step(args: list[str], work: Path, stage: str) -> Any:
    """Start a step as a background child; its output goes to _<stage>.log inside the work_dir."""
    import subprocess

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    log = open(str(work / f"_{stage}.log"), "w", encoding="utf-8", errors="replace")
    try:
        proc = subprocess.Popen(
            [sys.executable, *args], cwd=str(Path(__file__).resolve().parent),
            stdout=log, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    finally:
        log.close()
    _job_write(work, stage=stage, status="running", pid=int(proc.pid), started=time.time())
    return proc


def _step_log(work: Path, stage: str) -> str:
    try:
        return (work / f"_{stage}.log").read_text(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        return ""


def _running_payload(work: Path, job: dict[str, Any], stage: str) -> dict[str, Any]:
    """The answer when a step is still going: not an error, and complete enough to resume from."""
    started = float(job.get("started") or time.time())
    return {
        "ok": True, "done": False, "status": "running", "stage": stage,
        "work_dir": str(work), "pid": int(job.get("pid") or 0),
        "elapsed_s": round(time.time() - started, 1),
        "poll_with": "cdr_job_status",
        "next": (f"The {stage} step is still running and has deliberately been left running. Call "
                 f"cdr_job_status(work_dir=r'{work}') again in about 30-60 s until done=true. Do NOT "
                 f"start another vectorize call for this image and do NOT redo the earlier step."),
    }


def _wait_step(proc: Any, work: Path, stage: str, budget: float) -> dict[str, Any] | None:
    """Wait up to `budget` seconds. None = it finished (the caller reads the output it produced);
    otherwise a complete 'still running' payload the caller returns instead of timing out."""
    try:
        proc.wait(timeout=max(float(budget), 1.0))
    except Exception:  # noqa: BLE001 - still running, which is the case we handle
        pass
    if proc.poll() is not None:
        return None
    return _running_payload(work, _job_read(work), stage)


def _workflow(payload: dict[str, Any]) -> dict[str, Any]:
    """The first field of every answer: which workflow step this is, and what must happen before the
    result may be reported to the user as finished.

    "ok: true" alone cannot express "succeeded, but the job is only half done", and a caller that reads
    only that will report a half-finished conversion as complete. This says it in one place, in front.
    """
    wd = payload.get("work_dir") or ""
    if payload.get("status") == "running":
        return {"step": "running", "step_label": f"{payload.get('stage')} 仍在运行", "done": False,
                "must_do_next": [f"稍后调用 cdr_job_status(work_dir=r'{wd}') 取结果"],
                "note": "这不是失败：任务还在跑，结果会出现在同一个 work_dir 里。"}
    if payload.get("delivered") is True:
        left = payload.get("remaining_issues") or []
        return {"step": "delivered", "step_label": "交付已确认", "done": True,
                "must_do_next": (["把 remaining_issues 如实写进给用户的答复，并说明哪些适合在 CorelDRAW "
                                  "里手工收尾（字体、字号、丢失的空格）"] if left else ["可以如实报告完成"]),
                "note": (f"仍有 {len(left)} 项已知问题随交付一起保留。" if left else "没有遗留问题。")}
    if not payload.get("ok"):
        # A refusal from the delivery gate is not a malfunction: it is the workflow telling the caller
        # what is still open, and it must not read like a crash.
        if payload.get("delivered") is False and payload.get("still_open"):
            return {"step": "blocked", "step_label": "交付被拦下：还有未处理的项", "done": False,
                    "must_do_next": list(payload["still_open"]) + ["处理完后重新调用 cdr_finish(work_dir)"],
                    "note": "这不是故障。处理完 still_open 再交付；确实无法自动修的项用 "
                            "accept_remaining=true 并在给用户的答复里如实说明原因。"}
        return {"step": "failed", "step_label": "失败", "done": True, "must_do_next": [],
                "note": str(payload.get("error") or payload.get("step") or "见返回内容")}
    pending: list[str] = []
    # Once the caller has edited the labels, the L#/M# list has been dealt with - nagging about it
    # after the fix only teaches the caller to ignore the field.
    edited = bool(payload.get("labels_edited"))
    check = payload.get("labels_to_check") or []
    if check and not edited:
        line = f"看 review_sheet，核对 {len(check)} 个不确定标签（L#）"
        if len(check) >= UNCERTAIN_HINT_MIN and not payload.get("to_fill"):
            # Past a certain count, guessing reading-by-reading is the wrong trade: a wrong reading must
            # be hunted down, while a blank is visibly blank and one human pass fixes them all.
            line += (f"。数量偏多（{len(check)} 个），也可以改用 cdr_vectorize_build(work_dir, "
                     f"skip_uncertain=True)：只放有把握的字，其余留白并在 CDR 里用 TO FILL 图层标出，"
                     f"由人工填字")
        pending.append(line)
    if payload.get("unlabeled_text") and not edited:
        pending.append(f"确认 {len(payload['unlabeled_text'])} 处疑似漏识别的文字（M#）")
    if payload.get("graphics_issues"):
        pending.append(f"处理 {len(payload['graphics_issues'])} 条图形问题（graphics_issues：改 mode/colors/框）")
    if payload.get("text_issues"):
        pending.append(f"修正 {len(payload['text_issues'])} 条文字放置问题（text_issues：改 labels）")
    if pending:
        return {"step": "review", "step_label": "2/3 核对并重建", "done": False,
                "must_do_next": pending + [f"再调用 cdr_vectorize_build(work_dir=r'{wd}', label_patch=...) 重建",
                                           f"收敛后调用 cdr_finish(work_dir=r'{wd}') 才算交付"],
                "note": "一次调用只是第一遍，上面的问题不处理就等于只做了一半。"}
    return {"step": "converged", "step_label": "3/3 已收敛，可交付", "done": True,
            "must_do_next": [f"调用 cdr_finish(work_dir=r'{wd}') 取得交付确认和最终文件清单"],
            "note": "没有待处理的核对项。"}


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

# ---------------------------------------------------------------------------
# Tool results carry the pictures, not only their paths.
#
# "review_sheet": "C:\\...\\label_review.png" is a string. To act on it the calling AI has to work out
# on its own that it should open that file, and to have a way to do it - which is precisely why the
# review half of the workflow gets skipped. An MCP image block, by contrast, is in the caller's context
# whether it asked for it or not: the review sheet gets looked at because it is simply there.
# ---------------------------------------------------------------------------

INLINE_MAX_SIDE = 1280        # keep a full-page sheet inside a sane token budget
INLINE_MAX_BYTES = 700_000    # above this, re-encode as JPEG
INLINE_MAX_IMAGES = 2


def _inline_image(path: str, max_side: int = INLINE_MAX_SIDE) -> Any:
    """Downscale one result picture and return it as MCP image content, or None on any problem."""
    import io

    try:
        from PIL import Image as PILImage

        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        im = PILImage.open(p)
        if im.mode not in ('RGB', 'L'):
            im = im.convert('RGB')
        w, h = im.size
        if max(w, h) > max_side:
            s = max_side / float(max(w, h))
            im = im.resize((max(1, int(w * s)), max(1, int(h * s))), PILImage.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format='PNG', optimize=True)
        data, fmt = buf.getvalue(), 'png'
        if len(data) > INLINE_MAX_BYTES:
            buf = io.BytesIO()
            im.convert('RGB').save(buf, format='JPEG', quality=82, optimize=True)
            data, fmt = buf.getvalue(), 'jpeg'
        return Image(data=data, format=fmt)
    except Exception:  # noqa: BLE001 - the JSON payload still carries the path
        return None


def _inline_choice(payload: dict[str, Any]) -> list[str]:
    """Which pictures the caller must look at, in workflow order (at most INLINE_MAX_IMAGES)."""
    sc = payload.get('self_check') or {}
    pics: list[str] = []
    if payload.get('review_sheet'):
        pics.append(str(payload['review_sheet']))
    if payload.get('issues') and sc.get('diff_png'):
        pics.append(str(sc['diff_png']))
    if not pics and payload.get('png'):
        pics.append(str(payload['png']))
    return pics[:INLINE_MAX_IMAGES]


def _tool_result(payload: dict[str, Any], *, is_error: bool = False) -> Any:
    """One tool answer: the JSON payload with `workflow` in front, then the pictures to inspect.

    `workflow` leads so that a caller reading only the start of the answer still learns whether this is
    a finished job, a half-done one, or a running one. is_error=True is what the delivery gate uses: it
    reaches the model as a tool error, which cannot be quietly reported as success.
    """
    import json

    if "workflow" not in payload:
        payload = {"workflow": _workflow(payload), **payload}
    out: list[Any] = [TextContent(type='text', text=json.dumps(payload, ensure_ascii=False, indent=1))]
    for path in _inline_choice(payload):
        img = _inline_image(path)
        if img is not None:
            out.append(img)
    if is_error:
        from mcp_types import CallToolResult

        return CallToolResult(content=out, is_error=True)
    return out


def _collect_prepare(work: Path) -> dict[str, Any] | None:
    """Assemble the STEP-1 payload from what is on disk. None while prepare has not finished writing
    ocr_candidates.json - which is also how a handed-off run is recognised as complete."""
    import json

    meta_path = work / "ocr_candidates.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - still being written
        return None
    import cdr_vectorize
    return _shrink_payload({
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
    }, work)


# Read once, at process start: only a value the USER exported counts as a manual override. The scale
# this server works out for one image is written to the environment so the child processes inherit it,
# and that written value must not be mistaken for an override on the next call.
_USER_SR = os.environ.get("CDR_SR", "").strip()


def _set_scale_for(image_path: Path) -> int:
    """Pick the super-resolution scale for this image and export it to the child processes.

    Upscaling only helps a small or blurry input. On a large, already-crisp image the x2 upscale pushes
    the working size past MAX_WORK_MPX, and prepare then shrinks the SOURCE back down to fit - paying for
    super-resolution and immediately undoing it, at the cost of about 60% of the source pixels and half
    the glyph height. Measured on a 2048x1824 map (3.7 MP): the working image went 2594x2310 -> 2048x1824
    (smaller!), OCR-uncertain labels fell 31 -> 13, and the graphics check went from
    precision 0.666 / 23 lost elements to 1.0 / 0. Set CDR_SR to override the choice.
    """
    if _USER_SR:
        return max(1, int(_USER_SR))
    scale = 2
    try:
        from PIL import Image

        with Image.open(image_path) as im:
            w, h = im.size
        try:
            import cdr_vectorize

            cap = float(cdr_vectorize.MAX_WORK_MPX)
        except Exception:  # noqa: BLE001
            cap = 6.0
        # 4 is S=2 squared: the working image is the source times S in each direction
        if (w * h * 4) / 1e6 > cap:
            scale = 1
    except Exception:  # noqa: BLE001 - an unreadable size falls back to the historical default
        scale = 2
    os.environ["CDR_SR"] = str(scale)
    return scale


def _prepare_impl(image_path: str, work_dir: str = "",
                  wait_seconds: float = DEFAULT_WAIT_SECONDS) -> dict[str, Any]:
    """Run STEP 1 within a bounded wait; hand back a resumable handle when the budget runs out."""
    # absolute paths only: CorelDRAW resolves relative paths against its own working directory and
    # fails with a bare "发生意外" (unexpected error) on Import
    p = Path(image_path).resolve()
    if not p.exists():
        return {"ok": False, "error": f"image not found: {image_path}"}
    work = Path(work_dir).resolve() if work_dir else _ensure_workdir() / f"vec_{p.stem}_{int(time.time())}"
    work.mkdir(parents=True, exist_ok=True)
    scale = _set_scale_for(p)
    # a prepare that already finished is reused rather than redone: this is what makes cdr_job_status
    # cheap and what makes a resumed call safe. Re-run with a fresh work_dir to re-prepare a new image.
    already = _collect_prepare(work)
    if already is not None:
        _job_write(work, stage="prepare", status="done", mode=already.get("mode"))
        return already
    try:
        proc = _start_step(["cdr_vectorize.py", "prepare", str(p), str(work)], work, "prepare")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"prepare step failed to start: {exc}", "work_dir": str(work)}
    handed = _wait_step(proc, work, "prepare", wait_seconds)
    if handed is not None:
        return handed
    got = _collect_prepare(work)
    if got is None:
        err = _step_log(work, "prepare")
        _job_write(work, status="failed", error=err[-1500:])
        return {"ok": False, "error": (err or "prepare produced no result")[-1500:], "work_dir": str(work)}
    _job_write(work, stage="prepare", status="done", mode=got.get("mode"))
    return got


def _uncertain_candidates(meta: dict) -> dict[int, dict[str, Any]]:
    """{candidate index: review info} for every reading prepare flagged as uncertain.

    These are exactly the items the review sheet lists: the readings a model is asked to make out of a
    picture, and the ones it most often gets wrong.
    """
    out: dict[int, dict[str, Any]] = {}
    for i, c in enumerate(meta.get("candidates", [])):
        weak = not c.get("use", True) and str(c.get("skip_reason", "")).startswith("single weak")
        if (c.get("use", True) and c.get("uncertain")) or weak:
            alts: list[str] = []
            for r in c.get("readings", []):
                if r[0] not in alts and r[0] != c["text"]:
                    alts.append(r[0])
            out[i] = {"text": c["text"], "why": c.get("why") or c.get("skip_reason", ""),
                      "other_readings": alts[:3], "placed": bool(c.get("use", True))}
    return out


def _labels_with_skip(meta_prep: dict) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split the labels prepare selected into 'place it' and 'leave the spot blank for a human'.

    A wrong reading sitting on the page is worse than a blank one: it has to be found and corrected,
    while a blank is at least visibly blank. The blanked labels are still handed to the build so their
    boxes get wiped from the image - a text ghost left behind would be traced as linework - but they
    carry erase_only, so no text object is created for them.

    Candidates prepare already decided NOT to place are left completely alone: they may be symbols
    rather than text, so their ink must not be wiped either.
    """
    unc = _uncertain_candidates(meta_prep)
    keep: list[dict[str, Any]] = []
    blanks: list[dict[str, Any]] = []
    for i, c in enumerate(meta_prep.get("candidates", [])):
        if not c.get("use", True):
            continue
        text = str(c.get("text", "") or "").strip()
        if not text:
            continue
        split = c.get("split_boxes") or []
        parts = list(zip(text.split("|"), split)) if split and "|" in text else [(text, c["box"])]
        for part, box in parts:
            item = {"text": str(part), "box": [int(v) for v in box],
                    "angle": float(c.get("angle", 0.0) or 0.0)}
            if i in unc:
                blanks.append({**item, "erase_only": True, "id": f"L{i}",
                               "why": unc[i]["why"], "other_readings": unc[i]["other_readings"]})
            else:
                keep.append(item)
    return keep, blanks


def _label_index(lid: Any) -> int | None:
    """'L3' -> 3, the candidate the review sheet calls L3. 'M#' has no candidate index."""
    s = str(lid).strip()
    return int(s[1:]) if len(s) > 1 and s[0] in ('L', 'l') and s[1:].isdigit() else None


def _apply_label_patch(meta_prep: dict, patch: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the corrected label list from a patch keyed by the L#/M# ids the caller saw.

    Making a caller re-send a hundred labels to fix two is an O(n) edit for an O(1) change, and
    re-sending is itself where labels get truncated, reordered or mistyped - a correct list turning
    into a wrong one. A patch cannot damage what it does not touch.
    """
    import copy

    import cdr_vectorize

    cands = copy.deepcopy(meta_prep.get("candidates", []))
    for lid, new_text in (patch.get("set") or {}).items():
        i = _label_index(lid)
        if i is None or not (0 <= i < len(cands)):
            continue
        text = "" if new_text is None else str(new_text).strip()
        if not text:
            cands[i]["use"] = False          # empty reading = "not text, trace it as graphics"
        else:
            cands[i]["text"] = text
            cands[i]["use"] = True
            cands[i]["uncertain"] = False
            cands[i].pop("split_boxes", None)
    for lid in (patch.get("drop") or []):
        i = _label_index(lid)
        if i is not None and 0 <= i < len(cands):
            cands[i]["use"] = False
    for lid, box in (patch.get("set_box") or {}).items():
        i = _label_index(lid)
        if i is not None and 0 <= i < len(cands) and isinstance(box, (list, tuple)) and len(box) == 4:
            cands[i]["box"] = [int(v) for v in box]
    out = cdr_vectorize.suggested_labels({**meta_prep, "candidates": cands})
    for extra in (patch.get("add") or []):
        box = extra.get("box") if isinstance(extra, dict) else None
        if isinstance(extra, dict) and extra.get("text") and isinstance(box, (list, tuple)) and len(box) == 4:
            item = {"text": str(extra["text"]), "box": [int(v) for v in box]}
            if "angle" in extra:
                item["angle"] = float(extra["angle"])
            out.append(item)
    return out


def _carry_prep(result: dict[str, Any], work: Path) -> dict[str, Any]:
    """Merge step 1's findings into a finished build answer, so both ways of collecting a result -
    the one-call flow and cdr_job_status - hand the caller the same complete payload, including the
    L#/M# review list it still has to act on. Idempotent."""
    prep = _collect_prepare(work)
    if prep:
        for k in ("labels", "labels_to_check", "unlabeled_text", "review_sheet",
                  "source_png", "source_scale", "source_size"):
            if prep.get(k) is not None:
                result.setdefault(k, prep[k])
    return result


def _shrink_payload(payload: dict[str, Any], work: Path) -> dict[str, Any]:
    """Trim the long lists out of an answer.

    A 40 kB JSON body is what makes a client truncate a tool result - and the inline pictures, the
    whole reason the answer carries them, are the first thing to be lost. The full lists stay in the
    work_dir, and label_patch means the caller never needs to re-send them anyway. The review lists
    (labels_to_check / to_fill) are left intact: dropping those would hide the very items the caller
    is required to act on.
    """
    labels = payload.get("labels")
    if isinstance(labels, list) and len(labels) > MAX_LIST_ITEMS:
        payload["labels"] = labels[:MAX_LIST_ITEMS]
        payload["labels_total"] = len(labels)
        payload["labels_file"] = str(work / "labels_confirmed.json")
        payload["labels_note"] = (f"这里只列前 {MAX_LIST_ITEMS} 项（共 {len(labels)} 项）。改标签请用 "
                                  f"label_patch 按 L# 索引，不要重传整份列表；完整内容见 labels_file。")
    lp = payload.get("label_placement")
    if isinstance(lp, list) and len(lp) > MAX_LIST_ITEMS:
        payload["label_placement"] = lp[:MAX_LIST_ITEMS]
        payload["label_placement_total"] = len(lp)
        payload["label_placement_note"] = f"这里只列前 {MAX_LIST_ITEMS} 项（共 {len(lp)} 项）。"
    return payload


def _finish_build_payload(work: Path, result: dict[str, Any], is_color: bool) -> dict[str, Any]:
    """Add both self-checks and the pre-computed placement to a finished CorelDRAW result."""
    import json

    trace_stats = result.pop("layers_built", None)
    try:
        import cdr_selfcheck
        # CHECKPOINT first: it runs on the graphics-only export, so with no text on the page every
        # shortfall is a tracing fault. The final check cannot separate "a line was lost" from "a text
        # object was placed on top of that line" - here they arrive as two different reports.
        gfx = cdr_selfcheck.graphics_self_check(str(work), trace_stats)
        if gfx:
            result["graphics_issues"] = gfx["issues"]
            result["graphics_check"] = {**gfx["metrics"], "diff_png": gfx["diff_png"]}
            if gfx.get("spill"):
                result["wipe_spill"] = {
                    "px": gfx["spill"]["px"], "share": gfx["spill"]["share"],
                    "regions": gfx["spill"]["regions"],
                    "meaning": "linework outside every label box was wiped as text before tracing, so "
                               "it cannot appear in the result: a label box is too large or covers "
                               "linework, and only the box (not the text) can fix it",
                }
        chk = cdr_selfcheck.self_check(str(work), trace_stats)
        result["issues"] = chk["issues"]            # graphic problems for the caller to act on ([] = none)
        result["self_check"] = {**chk["metrics"], "diff_png": chk["diff_png"]}
        # Split the final issues by WHAT FIXES THEM: a text fault is a label edit, everything else is a
        # tracing / colour / mode decision. Merged into one list they look alike and get fixed alike.
        result["text_issues"] = [i for i in chk["issues"] if i.get("type") == "text_mismatch"]
        result["graphics_issues"] = (result.get("graphics_issues") or []) + [
            {**i, "found_after_text": True} for i in chk["issues"] if i.get("type") != "text_mismatch"]
    except Exception as exc:  # noqa: BLE001 - the check must never fail a finished build
        result["issues"] = [{"type": "self_check_error", "detail": f"{type(exc).__name__}: {exc}"}]
    result["mode"] = "color" if is_color else "lineart"
    result["work_dir"] = str(work)
    # whether the caller has already acted on the review list; reported so the workflow block and the
    # delivery gate stop asking for something that has been done
    result["labels_edited"] = bool(_job_read(work).get("labels_edited"))
    # The placement pre-computed for every label (size / angle / position / colour / faux-bold), read
    # back from the build metadata: the caller sees what the text objects were given, not only what the
    # finished page happens to look like. A wrong value here is a label edit, not a guess.
    try:
        lf = work / ("layers.json" if is_color else "labels.json")
        if lf.exists():
            m = json.loads(lf.read_text(encoding="utf-8"))
            sc = float(m.get("scale", 2)) or 2.0
            alll = m.get("labels", [])
            blanks = sum(1 for l in alll if l.get("erase_only"))
            unmeas = sum(1 for l in alll if l.get("unmeasurable"))
            # BOTH kinds end up without a text object, so both must come off the count: erase_only is
            # left for a human to type, and unmeasurable is left blank because its run could not be
            # measured. Counting only erase_only over-reported this by 26 labels on a real map.
            result["labels_placed"] = len(alll) - blanks - unmeas
            result["labels_erased_blank"] = blanks            # wiped, left empty for a human
            result["labels_unmeasurable"] = unmeas            # left blank, size could not be measured
            wp = int(m.get("wipe_protected_px") or 0)
            if wp:
                result["wipe_protected"] = {
                    "px": wp,
                    "meaning": "这些墨被保护下来没有擦掉，因为形状不像文字（线、剖面线、小符号压在标签框里）。"
                               "擦除发生在描摹之前，擦掉的东西交付时就永久缺失，所以这里偏向保留；"
                               "如果结果里出现文字残影，说明保护过头了。",
                }
            # box/tight are stored in working pixels (source x scale); report source pixels so they
            # line up with `labels` and with every region the checks report.
            result["label_placement"] = [
                {"text": l.get("text"),
                 "box": [round(v / sc, 1) for v in l.get("box", [])],
                 "tight": [round(v / sc, 1) for v in l.get("tight", [])],
                 "place": l.get("place")}
                for l in m.get("labels", []) if l.get("place")]
    except Exception:  # noqa: BLE001 - placement reporting is informational
        pass
    # Labels left blank for a human: their boxes were wiped (so no text ghost is traced) but no text
    # object was placed, and the CDR carries a magenta outline at each spot.
    #
    # The magenta boxes come from TWO causes: labels the caller chose to leave blank (skip_uncertain),
    # and slanted labels whose text run could not be measured. Only the first had a list, so
    # to_fill_marks could exceed to_fill_count with nothing explaining the difference - a caller seeing
    # 26 boxes and 0 listed items has no way to act. List both, in source pixels like everything else.
    tf = _job_read(work).get("to_fill") or []
    try:
        _lf = work / "layers.json"
        if _lf.exists():
            _m = json.loads(_lf.read_text(encoding="utf-8"))
            _sc = float(_m.get("scale", 2)) or 2.0
            for _l in _m.get("labels", []):
                if not _l.get("unmeasurable"):
                    continue
                tf.append({"text": _l.get("text") or "",
                           "box": [round(v / _sc, 1) for v in _l.get("box", [])],
                           "angle": _l.get("angle", 0.0),
                           "why": "斜排标签的文字长度测不出（字形被线或色块干扰），照框推字号会得到巨字，"
                                  "所以故意留空"})
    except Exception:  # noqa: BLE001 - reporting only
        pass
    if tf:
        result["to_fill"] = tf
        result["to_fill_count"] = len(tf)
        result["to_fill_note"] = ("这些位置在 CDR 里故意留空，由 TO FILL 图层上的品红框标出：双击框内直接打字，"
                                  "填完删掉该图层。每项的位置和原因见 to_fill 的 why 字段，可用来对照。"
                                  "它们不是失败，不需要重跑：读不准的读数和不适合自动定字号的斜排标签都走这条路。")
    return _carry_prep(result, work)


def _collect_build(work: Path, is_color: bool, use_cache: bool = True) -> dict[str, Any] | None:
    """Assemble the STEP-2 payload once the CorelDRAW step has written its JSON line (None before).

    Cached in _job.json because the self-checks behind it take seconds and cdr_job_status may be
    polled several times for the same finished build.
    """
    import json

    if use_cache:
        cached = _job_read(work).get("payload")
        if isinstance(cached, dict) and cached.get("ok"):
            return _shrink_payload(_carry_prep(cached, work), work)
    meta_file = "layers.json" if is_color else "labels.json"
    if not (work / meta_file).exists():
        return None
    lines = [ln for ln in _step_log(work, "coreldraw").splitlines() if ln.startswith("{")]
    if not lines:
        return None
    try:
        result = json.loads(lines[-1])
    except Exception:  # noqa: BLE001 - a partial or interleaved line
        return None
    if not result.get("ok"):
        result.setdefault("step", "coreldraw")
        result["work_dir"] = str(work)
        return result
    result = _finish_build_payload(work, result, is_color)
    if result.get("ok"):
        # the cache keeps the full lists; only the copy handed to the caller is trimmed
        _job_write(work, payload=result)
    return _shrink_payload(result, work)


def _build_impl(
    work_dir: str,
    labels: list[dict[str, Any]] | None = None,
    font: str = "新宋体",
    trace_type: str = "lineart",
    detail: int = 100,
    smoothing: int = 25,
    mode: str = "auto",
    colors: int = 12,
    label_patch: dict[str, Any] | None = None,
    skip_uncertain: bool = False,
    latin_font: str = "",
    wait_seconds: float = DEFAULT_WAIT_SECONDS,
) -> dict[str, Any]:
    """Run STEP 2 within a bounded wait; hand back a resumable handle when the budget runs out."""
    import json

    t_start = time.time()
    work = Path(work_dir).resolve()          # relative paths break CorelDRAW's Import (see prepare)
    if not (work / "sr2.png").exists() and not (work / "src.png").exists():
        return {"ok": False, "error": f"not a prepared work_dir (run cdr_vectorize_prepare first): {work_dir}"}
    job = _job_read(work)
    meta_prep: dict[str, Any] = {}
    if (work / "ocr_candidates.json").exists():
        meta_prep = json.loads((work / "ocr_candidates.json").read_text(encoding="utf-8"))
    # prepare recorded the super-resolution scale it used; every child process must run at the same one
    # or all box coordinates would be off by that factor
    try:
        _sc = int(meta_prep.get("scale", 2) or 2)
        if not os.environ.get("CDR_SR", "").strip():
            os.environ["CDR_SR"] = str(max(1, _sc))
    except Exception:  # noqa: BLE001
        pass
    if mode == "auto":
        mode = str(job.get("mode") or meta_prep.get("mode", "lineart"))
    is_color = mode == "color"

    # A rebuild was explicitly asked for (new labels, a patch, or the blank-out policy) - that must
    # actually rebuild rather than hand back the cached answer.
    edited = labels is not None or bool(label_patch) or skip_uncertain
    # Otherwise, when this work_dir already produced a result, hand the same answer back instead of
    # rebuilding: that is what makes cdr_job_status and a resumed call idempotent rather than a way to
    # burn CorelDRAW time twice.
    if not edited and str(job.get("status")) == "done" and str(job.get("stage")) == "coreldraw":
        again = _collect_build(work, is_color)
        if again is not None:
            return again

    to_fill: list[dict[str, Any]] = []
    if labels is None:
        if label_patch:
            labels = _apply_label_patch(meta_prep, label_patch)
        elif skip_uncertain:
            keep, to_fill = _labels_with_skip(meta_prep)
            # the blanks ride along so their boxes still get wiped, but erase_only keeps the text
            # object out of the build
            labels = keep + to_fill
        else:
            import cdr_vectorize
            labels = cdr_vectorize.suggested_labels(meta_prep)
    clean = []
    for lab in labels or []:
        box = lab.get("box")
        erase_only = bool(lab.get("erase_only"))
        if not isinstance(box, (list, tuple)) or len(box) != 4 or (not lab.get("text") and not erase_only):
            return {"ok": False, "error": f"each label needs text and box [x0,y0,x1,y1]: {lab}"}
        item = {"text": str(lab.get("text") or ""), "box": [int(v) for v in box]}
        if "angle" in lab:
            item["angle"] = float(lab["angle"])
        if erase_only:
            item["erase_only"] = True
        clean.append(item)
    confirmed = work / "labels_confirmed.json"
    meta_file = "layers.json" if is_color else "labels.json"
    # The layers step may already be done: the one-call flow hands back while it runs, and the status
    # call then comes here to finish the job. Re-running it threw that work away and, on a big drawing
    # where layers alone outlasts the wait budget, every round timed out in the same place until the
    # round limit was reached ("the CorelDRAW step was never started"). Reuse it when nothing about
    # the labels changed and the file is newer than the labels it was built from.
    reuse_layers = (not edited and confirmed.exists() and (work / meta_file).exists()
                    and (work / meta_file).stat().st_mtime >= confirmed.stat().st_mtime
                    and json.dumps(clean, ensure_ascii=False, indent=1)
                    == confirmed.read_text(encoding="utf-8"))
    if not reuse_layers:
        confirmed.write_text(json.dumps(clean, ensure_ascii=False, indent=1), encoding="utf-8")
    rounds = int(job.get("rounds") or 0) + 1
    # payload=None drops the cached previous result: a rebuild must never hand the old one back
    _job_write(work, mode=mode, rounds=rounds, labels_edited=edited, status="running",
               stage=("layers" if is_color else "finalize"), pid=0, started=time.time(), payload=None,
               to_fill=to_fill)

    if latin_font:
        # the serif/sans choice cannot be measured at drawing resolution (see detect_latin_font), so a
        # caller who can see the drawing may simply say which one to use
        os.environ["CDR_LATIN_FONT"] = str(latin_font)
    prep_args = (["cdr_vectorize_color.py", "layers", str(work), str(confirmed), str(colors)]
                 if is_color else ["cdr_vectorize.py", "finalize", str(work), str(confirmed)])
    stage_layers = "layers" if is_color else "finalize"
    if not reuse_layers:
        try:
            proc = _start_step(prep_args, work, stage_layers)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{stage_layers} step failed to start: {exc}",
                    "work_dir": str(work)}
        handed = _wait_step(proc, work, stage_layers, min(float(wait_seconds), 120.0))
        if handed is not None:
            return handed
    meta_path = work / meta_file
    # A crash in this step leaves the PREVIOUS run's file in place, and testing only for existence then
    # uses that stale file silently - the build "succeeds" on last run's data. Require it to be newer
    # than the confirmed labels written moments ago.
    fresh = meta_path.exists() and meta_path.stat().st_mtime >= confirmed.stat().st_mtime
    if not fresh:
        err = _step_log(work, stage_layers)
        _job_write(work, status="failed", error=err[-1500:])
        return {"ok": False, "step": stage_layers, "work_dir": str(work),
                "error": (err or f"{stage_layers} produced no fresh {meta_file} "
                                 "(a stale file from an earlier run was left in place)")[-1500:]}

    # A hung instance answers nothing, and the old readiness test used a COM call with no timeout - so
    # the check hung on exactly the case it was meant to catch. ensure_cdr_healthy asks in a child
    # process, and restarts an instance that is unresponsive or has burnt enough CPU to be about to be.
    health = ensure_cdr_healthy()
    if not _cdr_ping(timeout=15):
        return {"ok": False, "step": "coreldraw", "work_dir": str(work),
                # The server starts CorelDRAW itself from CDR_EXE, so the usual reason it cannot is
                # that CorelDRAW is installed somewhere else - and the old message never said so,
                # leaving the caller to guess (or to read the README).
                "error": ("CorelDRAW is not running and could not be started from "
                          f"{CDR_EXE}. Open CorelDRAW 2022 yourself, or set the CDR_EXE "
                          "environment variable to its CorelDRW.exe, then call this again."),
                "cdr_exe": CDR_EXE}
    com_args = ["cdr_vectorize_com.py", str(work), font, trace_type, str(detail), str(smoothing)]
    if is_color:
        com_args.append("color")
    result: dict[str, Any] | None = None
    for attempt in range(2):
        remaining = max(float(wait_seconds) - (time.time() - t_start), 5.0)
        proc = _start_step(com_args, work, "coreldraw")
        handed = _wait_step(proc, work, "coreldraw", remaining)
        if handed is not None:
            return handed
        result = _collect_build(work, is_color, use_cache=not edited)
        if isinstance(result, dict) and health.startswith("restarted"):
            result["cdr_restarted"] = health
        if result is None or result.get("ok") or attempt:
            break
        err = str(result.get("error", ""))
        if "timed out" in err:
            # An instance that has been building all session slows down until a step that takes 84 s
            # in a fresh CorelDRAW runs past its budget (measured on a 3295x1820 grid map: 84 s
            # fresh, over 415 s in one that had run all evening). The timeout also killed the step
            # mid-COM, so a half-built document is still open in there. Start the step again on a
            # new instance rather than reporting a failure the caller cannot act on.
            _kill_cdr()
            _launch_once()
            for _ in range(20):
                if _cdr_ping(timeout=15):
                    break
                time.sleep(3)
            health = "restarted (com timeout)"
            continue
        if "com_error" not in err:
            break
        # CorelDRAW intermittently rejects a COM call while busy ("发生意外"); the identical call
        # succeeds a few seconds later, so retry once before reporting a failure
        time.sleep(5)
    if result is None:
        return {"ok": False, "step": "coreldraw", "work_dir": str(work),
                "error": _step_log(work, "coreldraw")[-1500:] or "the CorelDRAW step produced nothing"}
    if not result.get("ok"):
        return result
    _job_write(work, stage="coreldraw", status="done", rounds=rounds, mode=mode)
    return result


def _vectorize_impl(image_path: str, work_dir: str = "", mode: str = "auto", font: str = "新宋体",
                    wait_seconds: float = DEFAULT_WAIT_SECONDS) -> dict[str, Any]:
    """The whole one-call flow sharing one time budget, so the two steps cannot each outlast the wait."""
    t_start = time.time()
    prep = _prepare_impl(image_path, work_dir, wait_seconds=wait_seconds)
    if not prep.get("ok") and prep.get("status") != "running":
        return prep
    # Whatever step ends up being handed off, this work_dir owes the rest of the one-call flow.
    # Recording it here - not only on the prepare handoff - is what lets cdr_job_status carry the job
    # to the end when the BUILD step is the one that runs out of budget. (Setting it only for prepare
    # left a layers-stage handoff stuck: nothing ever started the CorelDRAW step.)
    try:
        _job_write(Path(prep["work_dir"]), auto_continue=True)
    except Exception:  # noqa: BLE001 - the handle is still usable without it
        pass
    if prep.get("status") == "running":
        return prep
    left = max(float(wait_seconds) - (time.time() - t_start), 10.0)
    build = _build_impl(prep["work_dir"], None, font=font, mode=mode, wait_seconds=left)
    if not build.get("ok") or build.get("status") == "running":
        build["work_dir"] = prep["work_dir"]
        # carry step 1's findings into the handed-off answer, so the caller does not have to re-read
        # them - and so cdr_job_status can still report them once the build finishes
        for k in ("labels", "labels_to_check", "unlabeled_text", "review_sheet"):
            build.setdefault(k, prep.get(k))
        return build
    needs_review = bool(prep["labels_to_check"] or prep["unlabeled_text"] or build.get("issues")
                        or build.get("graphics_issues"))
    payload = {
        **build,
        "labels": prep["labels"],
        "labels_to_check": prep["labels_to_check"],
        "unlabeled_text": prep["unlabeled_text"],
        "review_sheet": prep["review_sheet"],
        "needs_review": needs_review,
        "next": ((("Tracing problems: work through graphics_issues and look at self_check.diff_png. "
                   if build.get("graphics_issues") else "")
                  + ("Text objects are placed wrong: see text_issues and label_placement. "
                     if build.get("text_issues") else "")
                  + (_FIX_HINT if (prep["labels_to_check"] or prep["unlabeled_text"]) else ""))
                 or "No uncertain items.")
                + " Compare result.png (attached) with the source before reporting success.",
    }
    return _shrink_payload(payload, Path(prep["work_dir"]))


def _job_status_impl(work_dir: str) -> dict[str, Any]:
    """Where a handed-off call stands, rebuilt from the work_dir so it also survives a restart."""
    work = Path(work_dir).resolve()
    if not work.exists():
        return {"ok": False, "done": True, "error": f"work_dir not found: {work_dir}"}
    job = _job_read(work)
    stage = str(job.get("stage") or "")
    is_color = str(job.get("mode") or "") == "color"
    if not is_color and (work / "layers.json").exists() and not (work / "labels.json").exists():
        is_color = True

    built = _collect_build(work, is_color)
    if built is not None and built.get("ok"):
        _job_write(work, status="done")
        return built

    # something we started is still running: say so, and let the caller poll again
    if _pid_alive(int(job.get("pid") or 0)):
        return _running_payload(work, job, stage or "working")

    # The one-call flow was cut between its two steps (its prepare handed off, or its layers step
    # ended). Finish it here, rather than making the caller reconstruct the pipeline's shape.
    prep = _collect_prepare(work)
    # Not when the CorelDRAW step itself is what ended: then there is nothing left to hand over to,
    # only a failure to report. Continuing here silently rebuilt on the same broken CorelDRAW, the
    # second round died on its first COM call ("FontList does not support enumeration") and its log
    # overwrote the first round's - so the caller saw a meaningless error and the real one was gone.
    if (prep is not None and job.get("auto_continue") and stage != "coreldraw"
            and int(job.get("rounds") or 0) < MAX_BUILD_ROUNDS):
        _job_write(work, stage="prepare", status="done", mode=prep.get("mode"))
        return _build_impl(str(work), None, mode=str(job.get("mode") or "auto"))

    meta_file = "layers.json" if is_color else "labels.json"
    if stage == "coreldraw" or (work / meta_file).exists():
        started = (work / "_coreldraw.log").exists()
        return {"ok": False, "done": True, "step": "coreldraw", "work_dir": str(work),
                "error": ("the CorelDRAW step stopped without producing a result" if started
                          else "the CorelDRAW step was never started"),
                "log_tail": _step_log(work, "coreldraw")[-1200:],
                "next": f"Call cdr_vectorize_build(work_dir=r'{work}') to continue with the CorelDRAW step."}

    if prep is not None:
        _job_write(work, status="done", stage="prepare", mode=prep.get("mode"))
        return prep
    if stage:
        return {"ok": False, "done": True, "step": stage, "work_dir": str(work),
                "error": f"the {stage} step stopped without producing its result",
                "log_tail": _step_log(work, stage)[-1200:],
                "next": "Call the tool that started it again."}
    return {"ok": False, "done": True, "work_dir": str(work),
            "error": f"no job recorded in {work_dir}"}


def _finish_impl(work_dir: str, accept_remaining: bool = False, note: str = "") -> dict[str, Any]:
    """Decide whether a result may be handed to the user, and if not, say exactly what is still open.

    Reporting a half-reviewed conversion as done is the failure mode this guards: "ok: true" from the
    build step is a statement about tracing, not about the review that the workflow requires. A check
    the caller cannot skip is worth more than a paragraph asking it to be careful.
    """
    work = Path(work_dir).resolve()
    if not work.exists():
        return {"ok": False, "delivered": False, "error": f"work_dir not found: {work_dir}"}
    job = _job_read(work)
    is_color = str(job.get("mode") or "") == "color"
    if not is_color and (work / "layers.json").exists() and not (work / "labels.json").exists():
        is_color = True
    built = _collect_build(work, is_color)
    if built is None or not built.get("ok"):
        left = max(0.0, time.time() - float(job.get("started") or time.time()))
        return {"ok": False, "delivered": False, "work_dir": str(work),
                "still_open": ["还没有可交付的结果"],
                "job": {"stage": job.get("stage"), "status": job.get("status"),
                        "elapsed_s": round(left, 1)},
                "next": (f"先取结果：cdr_job_status(work_dir=r'{work}')；如果还在跑就等它跑完，"
                         f"跑完了再回来 finish。")}

    prep = _collect_prepare(work) or {}
    rounds = int(job.get("rounds") or 0)
    edited = bool(job.get("labels_edited"))
    check = prep.get("labels_to_check") or []
    unlabeled = prep.get("unlabeled_text") or []
    gissues = built.get("graphics_issues") or []
    tissues = built.get("text_issues") or []
    open_items: list[str] = []
    if check and not edited:
        open_items.append(f"{len(check)} 个不确定标签（L#）从未核对：看 review_sheet，改 text 或删掉该标签")
    if unlabeled and not edited:
        open_items.append(f"{len(unlabeled)} 处疑似漏识别的文字（M#）从未确认：是文字就补进 labels")
    if gissues:
        kinds = "、".join(sorted({str(i.get('type')) for i in gissues}))
        open_items.append(f"{len(gissues)} 条图形问题未处理（{kinds}）：换 mode/colors 或改框后重建")
    if tissues:
        open_items.append(f"{len(tissues)} 条文字放置问题未处理（text_issues）：改 labels 后重建")
    exhausted = rounds >= MAX_BUILD_ROUNDS
    if open_items and not accept_remaining and not exhausted:
        return {"ok": False, "delivered": False, "work_dir": str(work), "rounds": rounds,
                "still_open": open_items,
                "next": ("不能作为完成交付：上面这些还没处理。用 cdr_vectorize_build(work_dir, "
                         "label_patch={'set': {...}, 'drop': [...], 'add': [...]}) 改标签重建，"
                         "图形类问题换 mode/colors 重建。若确认剩下的确实无法自动修（例如点状填充被"
                         "误判成文字），再调用 cdr_finish(accept_remaining=True, note='原因')，"
                         "并在给用户的答复里如实列出这些问题。"),
                }
    return {
        "ok": True, "delivered": True, "work_dir": str(work), "rounds": rounds,
        "result_cdr": built.get("cdr"), "result_png": built.get("png"),
        "labels_placed": built.get("labels_placed"),
        "accepted_remaining": bool(open_items),
        "remaining_issues": open_items,
        "must_report_to_user": ("把 remaining_issues 如实写进给用户的答复，并说明哪些适合在 CorelDRAW "
                                "里手工收尾（字体、字号、丢失的空格）" if open_items
                                else "没有遗留问题，可以如实报告完成"),
        "note": note or "",
    }


# ---------------------------------------------------------------------------
# Tools. Each of these returns its JSON payload AND, as real MCP image content, the pictures the
# caller is told to inspect - so looking at them happens because the images are in context, not
# because a path string in a field was noticed. structured_output=False keeps the mixed
# text+image result unstructured (see MCPServer.convert_result).
# ---------------------------------------------------------------------------


@mcp.tool(structured_output=False)
def cdr_vectorize_prepare(image_path: str, work_dir: str = "",
                          wait_seconds: float = DEFAULT_WAIT_SECONDS) -> Any:
    """STEP 1 of image -> editable CDR (line drawings, schematics, colour maps with labels; blurry,
    small or JPEG input is fine).

    Runs AI super-resolution for the graphics and multi-scale OCR for the labels, then decides on
    its own which OCR results become text objects. This is the cheap half; most callers want
    cdr_vectorize, which runs both steps.

    Returns, together with the pictures to look at:
      labels            ready-to-use list for cdr_vectorize_build (no review needed to proceed)
      mode              'color' or 'lineart', detected from the image
      labels_to_check   uncertain readings (L#), with the reason and alternative readings
      unlabeled_text    glyph-like ink no label covers (M#) - possibly text OCR missed
      review_sheet      ONE image showing every L#/M# crop with its current reading (attached)
    Boxes are [x0, y0, x1, y1] in pixels of source_png (large inputs are shrunk first: source_scale).

    wait_seconds bounds how long this call blocks. If the step is not done by then it returns
    done=false with the same work_dir instead of failing, and cdr_job_status(work_dir) picks it up.
    Keep it slightly under your MCP client's tool timeout (60 s by default)."""
    return _tool_result(_prepare_impl(image_path, work_dir, wait_seconds=wait_seconds))


@mcp.tool(structured_output=False)
def cdr_vectorize_build(
    work_dir: str,
    labels: list[dict[str, Any]] | None = None,
    font: str = "新宋体",
    trace_type: str = "lineart",
    detail: int = 100,
    smoothing: int = 25,
    mode: str = "auto",
    colors: int = 12,
    label_patch: dict[str, Any] | None = None,
    skip_uncertain: bool = False,
    latin_font: str = "",
    wait_seconds: float = DEFAULT_WAIT_SECONDS,
) -> Any:
    """STEP 2 of image -> editable CDR: rebuild from corrected labels. Needs a running CorelDRAW 2022
    and a work_dir from cdr_vectorize / cdr_vectorize_prepare. Super-resolution and OCR are reused;
    only layers, trace and text are rebuilt, so it takes about a minute.

    PREFER label_patch OVER labels. Fixing two labels should not mean re-sending a hundred - and
    re-sending is itself where entries get truncated, reordered or mistyped, turning a correct list
    into a wrong one:
      label_patch = {"set": {"L3": "藏"},          # new text for that L# ("" = not text, trace it)
                     "drop": ["L7"],               # remove this label
                     "set_box": {"L5": [x0,y0,x1,y1]},
                     "add": [{"text": "南", "box": [256,398,269,412], "angle": -90}]}
      labels      = the full corrected list, only when you are rewriting most of it

    EDIT `text` FIRST, and touch `box` only when the box itself is wrong. A box also decides which
    pixels are wiped from the image before it is traced, so a box that is too large deletes linework
    that nothing can bring back. `wipe_spill` reports exactly that, and only shrinking or dropping
    that label repairs it - changing its text does not.

    skip_uncertain=True: place ONLY the readings prepare was confident about and leave the uncertain
    ones as deliberate blanks. Use it when there are too many uncertain readings for one-by-one
    guessing to be worth it - a wrong reading has to be hunted down and corrected, while a blank is
    visibly blank and one human pass fixes them all. Those boxes are still wiped from the image (a
    leftover ghost would otherwise be traced as linework) but no text object is created; the CDR gets
    magenta outlines on a layer named "TO FILL (delete after typing)", and `to_fill` in the answer
    lists every spot with its OCR candidates for reference. Note the trade-off: a wipe that clips real
    linework costs graphics with nothing left to restore it from, so the wipe is bounded to
    glyph-shaped ink and whatever it protected is reported as `wipe_protected`.

    Returns result.cdr / result.png, the pictures to look at, and:
      graphics_issues   tracing faults; fixed with mode / colors, never by editing labels
      text_issues       text objects placed wrong (overlap, size, missing glyphs); fixed in `labels`
      label_placement   the size / angle / position / colour each label was given
      issues            the merged list, for compatibility
    mode: 'auto' (default, from the image), 'lineart' (black linework + text) or 'color' (flat colour
    layers + grey graticule + black ink + text). `colors` = palette size for colour mode.
    font: '新宋体' / '仿宋' / '黑体' / '微软雅黑'.

    wait_seconds bounds the blocking wait; running out of it returns done=false plus a work_dir, not
    an error - continue with cdr_job_status(work_dir)."""
    return _tool_result(_build_impl(work_dir, labels, font, trace_type, detail, smoothing, mode, colors,
                                    label_patch=label_patch, skip_uncertain=skip_uncertain,
                                    latin_font=latin_font,
                                    wait_seconds=wait_seconds))


@mcp.tool(structured_output=False)
def cdr_vectorize(image_path: str, work_dir: str = "", mode: str = "auto", font: str = "新宋体",
                  wait_seconds: float = DEFAULT_WAIT_SECONDS) -> Any:
    """ONE CALL image -> editable CorelDRAW file (line drawings, schematics, colour maps with text;
    blurry / small / JPEG input is fine). Needs a running CorelDRAW 2022.

    This is the ONLY correct entry point for "turn this picture into a CDR". Do not hand-draw the
    content with cdr_draw_* , and do not fall back to cdr_trace_image: that traces the shapes but
    cannot rebuild the text, so it is a different and much worse result.

    Super-resolves and traces the graphics, OCRs the labels and re-creates them as editable text,
    picks colour vs line-art mode itself, saves result.cdr + result.png, and returns two pictures you
    must look at: review_sheet (every uncertain label) and the diff image (red = in the source but
    missing, blue = drawn but not in the source).

    Read `workflow` FIRST - it is the first field and it tells you whether this is a finished job, a
    half-done one, or one that is still running. Then, when the job needs review:
      labels_to_check   L# items: uncertain readings, with `why` and `other_readings`
      unlabeled_text    M# items: glyph-like ink no label covers - possibly text OCR missed
      graphics_issues   tracing faults: fix with mode / colors, not by editing labels
      text_issues       text objects placed wrong: fix text/box in `labels`
      label_placement   the size / angle / position / colour each label was given - what you edit
      wipe_spill        linework destroyed by an oversized label box; only the box can repair it

    THE FIRST CALL IS NOT THE FINISHED JOB. Then:
      a. fix the labels with cdr_vectorize_build(work_dir, label_patch={...}) - super-resolution and
         OCR are reused, only layers/trace/text are rebuilt;
      b. repeat until issues is empty or only font/size detail is left, and at most 3 rounds;
      c. call cdr_finish(work_dir) - it refuses to confirm delivery while anything above is unhandled,
         so this is what makes the review actually happen rather than merely being recommended.

    If the answer comes back with done=false and status="running", the job is still going: call
    cdr_job_status(work_dir) again in 30-60 s. That is not a failure and nothing was lost."""
    return _tool_result(_vectorize_impl(image_path, work_dir, mode, font, wait_seconds=wait_seconds))


@mcp.tool(structured_output=False)
def cdr_job_status(work_dir: str) -> Any:
    """Continue a call that handed back a resumable handle, or re-read a finished one.

    Any of the vectorize tools may answer with done=false and status="running" when it runs out of its
    `wait_seconds` budget: the work is still going in the background and this is how you collect it.
    Call it with the same work_dir every 30-60 s. done=true means the answer is complete, and the
    payload is then exactly what the original tool call would have returned.

    It is safe and cheap to call repeatedly - a finished build is read back from the work_dir rather
    than rebuilt - and it works even after the server was restarted."""
    return _tool_result(_job_status_impl(work_dir))


@mcp.tool(structured_output=False)
def cdr_finish(work_dir: str, accept_remaining: bool = False, note: str = "") -> Any:
    """MANDATORY LAST STEP: confirm delivery of a conversion, and learn what is still open.

    Call this once you believe the job is done. It checks the review the workflow requires and REFUSES
    (delivered=false, reported to you as a tool error) while uncertain labels (L#), unrecognised text
    (M#), graphics_issues or text_issues are still unhandled - so a half-reviewed conversion cannot be
    presented to the user as finished.

    When it delivers, read `remaining_issues` and write them into your answer to the user honestly,
    together with what is better finished by hand in CorelDRAW (font, size, spaces lost by OCR).

    accept_remaining=true is the escape hatch for damage that genuinely cannot be repaired
    automatically (a stipple fill misread as text, say): it delivers, but you must still report the
    items, and `note` should say why they were left."""
    payload = _finish_impl(work_dir, accept_remaining, note)
    return _tool_result(payload, is_error=not payload.get("delivered"))


if __name__ == "__main__":
    mcp.run(transport="stdio")
