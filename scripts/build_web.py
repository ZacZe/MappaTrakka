"""Build the in-browser version of MappaTrakka into a static folder.

    python scripts/build_web.py              # -> dist/
    python scripts/build_web.py --serve      # build, then serve on localhost:8000

The output is plain static files, ready for GitHub Pages or any web host:
the page from web/, the mappatrakka package zipped up for Pyodide, and a
sample heatmap rendered from demo data for the front page.

Pyodide needs the page to come over http(s), so opening dist/index.html
straight from disk won't work. Use --serve to try it locally.
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import io
import logging
import shutil
import sys
import tempfile
import zipfile
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TEXT_FILES = {".html", ".js", ".css"}


def package_zip() -> bytes:
    """The mappatrakka package as a zip (Python sources only)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted((ROOT / "mappatrakka").rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            # Fixed timestamps so the same source gives the same zip.
            info = zipfile.ZipInfo(rel, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, path.read_bytes())
    return buf.getvalue()


def render_sample(dest: Path, size: int = 800) -> None:
    from PIL import Image

    from mappatrakka.config import Config
    from mappatrakka.demo import build_demo_export
    from mappatrakka.pipeline import run
    from mappatrakka.render import shrink

    with tempfile.TemporaryDirectory() as tmp:
        export = build_demo_export(Path(tmp) / "export")
        cfg = Config()
        cfg.render.width = cfg.render.height = size * 2
        cfg.output.path = str(Path(tmp) / "sample.png")
        cfg.output.use_cache = False
        cfg.output.workers = 1
        run(export, cfg, show_progress=False)
        with Image.open(cfg.output.path) as img:
            shrink(img.convert("RGB"), size).save(dest, quality=88, optimize=True)


def build(out: Path, sample: bool = True) -> Path:
    web = ROOT / "web"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    pkg = package_zip()
    digest = hashlib.sha256(pkg)
    for path in sorted(web.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(web).as_posix().encode())
            digest.update(path.read_bytes())
    build_id = digest.hexdigest()[:12]

    for path in sorted(web.rglob("*")):
        if not path.is_file():
            continue
        target = out / path.relative_to(web)
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in TEXT_FILES:
            text = path.read_text(encoding="utf-8").replace("__BUILD__", build_id)
            target.write_text(text, encoding="utf-8")
        else:
            shutil.copyfile(path, target)

    (out / "py").mkdir()
    (out / "py" / "mappatrakka.zip").write_bytes(pkg)
    # Tell GitHub Pages not to run the folder through Jekyll.
    (out / ".nojekyll").write_text("", encoding="utf-8")

    if sample:
        (out / "assets").mkdir(exist_ok=True)
        render_sample(out / "assets" / "sample.jpg")
    print(f"Built {out} (build {build_id})")
    return out


def serve(folder: Path, port: int) -> None:
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(folder))
    handler.extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript"}
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        print(f"Serving on http://127.0.0.1:{port}/  (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ROOT / "dist"), help="output folder (default dist/)")
    p.add_argument("--no-sample", action="store_true", help="skip rendering the front page sample image")
    p.add_argument("--serve", action="store_true", help="serve the result on localhost afterwards")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)
    out = build(Path(args.out), sample=not args.no_sample)
    if args.serve:
        serve(out, args.port)


if __name__ == "__main__":
    main()
