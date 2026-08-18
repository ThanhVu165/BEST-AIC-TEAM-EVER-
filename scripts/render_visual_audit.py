"""Turn an existing e2e_audit.html into a visual, frame-inspection report.

Usage:
  python scripts/render_visual_audit.py reports/e2e_audit.html
  python scripts/render_visual_audit.py reports/e2e_audit.html --output reports/e2e_visual_audit.html

The script extracts the Video/Frame rows already produced by e2e_audit.py,
reads those exact source frames from the local AIC videos, writes compact JPEG
thumbnails, and builds an HTML report with clickable full-size images.
"""
from __future__ import annotations

import argparse
import base64
import html
import re
import sqlite3
from pathlib import Path
from urllib.parse import unquote

import cv2

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "database" / "aic2026.sqlite"


def _rows_from_table(table_html: str) -> list[list[str]]:
    rows = []
    for raw_row in re.findall(r"<tr>(.*?)</tr>", table_html, flags=re.S | re.I):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", raw_row, flags=re.S | re.I)
        clean = [re.sub(r"<[^>]+>", "", cell).strip() for cell in cells]
        if clean:
            rows.append([html.unescape(x) for x in clean])
    return rows


def _tables(page: str) -> list[list[list[str]]]:
    return [_rows_from_table(x) for x in re.findall(r"<table[^>]*>(.*?)</table>", page, flags=re.S | re.I)]


def _video_paths(db: Path) -> dict[str, Path]:
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute("SELECT video_id, path FROM videos").fetchall()
    finally:
        conn.close()
    result: dict[str, Path] = {}
    for video_id, stored in rows:
        p = Path(str(stored))
        candidates = [p if p.is_absolute() else ROOT / p, p if p.is_absolute() else db.parent / p]
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved.is_file():
                result[str(video_id)] = resolved
                break
    return result


def _thumb(video_path: Path, frame_id: int, out: Path) -> tuple[bool, str]:
    out.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return False, "video open failed"
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id)):
            return False, "seek failed"
        ok, frame = cap.read()
        if not ok or frame is None:
            return False, "frame read failed"
        h, w = frame.shape[:2]
        max_w = 480
        if w > max_w:
            frame = cv2.resize(frame, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return True, ""
    finally:
        cap.release()


def _data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def _build_cards(rows: list[list[str]], video_paths: dict[str, Path], assets: Path, prefix: str) -> tuple[str, int]:
    if not rows:
        return "<p>No rows.</p>", 0
    header = rows[0]
    try:
        video_i = header.index("Video")
        frame_i = header.index("Frame")
    except ValueError:
        return "<p>Could not find Video/Frame columns.</p>", 0

    cards = []
    generated = 0
    for row in rows[1:]:
        if len(row) <= max(video_i, frame_i):
            continue
        video_id = row[video_i]
        try:
            frame_id = int(float(row[frame_i]))
        except ValueError:
            continue
        video = video_paths.get(video_id)
        thumb_path = assets / f"{prefix}_{video_id}_{frame_id}.jpg"
        ok = False
        error = "video not found"
        if video:
            if thumb_path.is_file():
                ok, error = True, ""
            else:
                ok, error = _thumb(video, frame_id, thumb_path)
        if ok:
            generated += 1
            src = _data_uri(thumb_path)
            image = f'<a href="{src}" target="_blank"><img src="{src}" loading="lazy" alt="{html.escape(video_id)} frame {frame_id}"></a>'
        else:
            image = f'<div class="missing">{html.escape(error)}</div>'
        meta = "".join(f"<div><b>{html.escape(header[i])}</b>: {html.escape(row[i])}</div>" for i in range(min(len(header), len(row))))
        cards.append(f'<article class="card"><div class="image">{image}</div><div class="meta">{meta}</div></article>')
    return "<section class='cards'>" + "".join(cards) + "</section>", generated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("html_report", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--max-retrieval", type=int, default=50)
    parser.add_argument("--max-top100", type=int, default=100)
    args = parser.parse_args()

    source = args.html_report.read_text(encoding="utf-8")
    tables = _tables(source)
    if len(tables) < 3:
        raise SystemExit("Expected the standard e2e_audit.html with retrieval, decoding and Top-100 tables.")

    retrieval_rows = tables[1][: args.max_retrieval + 1]
    top_rows = tables[3][: args.max_top100 + 1] if len(tables) > 3 else []
    videos = _video_paths(args.db)

    output = args.output or args.html_report.with_name("e2e_visual_audit.html")
    assets = output.parent / (output.stem + "_assets")
    retrieval_html, n1 = _build_cards(retrieval_rows, videos, assets, "retrieval")
    top_html, n2 = _build_cards(top_rows, videos, assets, "top100")

    title = "AIC2026 Visual E2E Audit"
    page = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>{title}</title><style>
body{{margin:0;background:#0e1014;color:#e8eaed;font:14px Segoe UI,Arial,sans-serif}}main{{max-width:1700px;margin:auto;padding:24px}}h1{{margin:0 0 6px}}h2{{margin-top:34px}}.note{{color:#aeb4bf;margin-bottom:20px}}.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:16px}}.card{{background:#181b21;border:1px solid #2b3039;border-radius:12px;overflow:hidden}}.image{{background:#090a0d;min-height:190px;display:flex;align-items:center;justify-content:center}}img{{display:block;width:100%;height:auto;cursor:zoom-in}}.meta{{padding:12px;line-height:1.55}}.meta div{{border-bottom:1px solid #292e36;padding:2px 0}}.missing{{padding:30px;color:#ff7b7b}}details{{margin:18px 0}}summary{{cursor:pointer;font-size:18px;font-weight:600}}
</style></head><body><main><h1>{title}</h1><div class='note'>Images are the exact source-video frames referenced by the audit. Click any image to open the full thumbnail. Retrieval: {n1} images. Top-100: {n2} images.</div>
<details open><summary>Retrieval candidates — manual inspection</summary>{retrieval_html}</details>
<details open><summary>Final Top-100 — manual inspection</summary>{top_html}</details>
</main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    print(f"Visual HTML: {output}")
    print(f"Images generated: {n1 + n2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
