"""Generate every favicon / PWA / OG asset from the master SVGs.

Idempotent: re-run whenever you edit a brand SVG. Output lives under
``packages/cockpit/web/static/brand/`` so the templates can ``<link rel>`` them.

Run with the venv:
    .venv/bin/python tools/build_icons.py
"""

from __future__ import annotations

import io
from pathlib import Path

import cairosvg
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
BRAND_DIR = REPO_ROOT / "packages" / "cockpit" / "web" / "static" / "brand"

MASTER = BRAND_DIR / "logo.svg"
OG_MASTER = BRAND_DIR / "og-image.svg"


def render(svg_path: Path, png_path: Path, size: int | tuple[int, int]) -> None:
    if isinstance(size, int):
        w = h = size
    else:
        w, h = size
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=str(png_path),
        output_width=w,
        output_height=h,
    )
    print(f"  wrote {png_path.relative_to(REPO_ROOT)} ({w}x{h}, {png_path.stat().st_size:,} B)")


def build_ico() -> None:
    """Multi-size .ico containing 16/32/48 (Windows pinned tab + legacy)."""
    sizes = [(16, 16), (32, 32), (48, 48)]
    images: list[Image.Image] = []
    for w, h in sizes:
        buf = io.BytesIO()
        cairosvg.svg2png(url=str(MASTER), write_to=buf, output_width=w, output_height=h)
        buf.seek(0)
        images.append(Image.open(buf).convert("RGBA"))
    ico_path = BRAND_DIR / "favicon.ico"
    images[0].save(ico_path, format="ICO", sizes=sizes, append_images=images[1:])
    print(f"  wrote {ico_path.relative_to(REPO_ROOT)} (multi-size, {ico_path.stat().st_size:,} B)")


def main() -> None:
    BRAND_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building icons from {MASTER.relative_to(REPO_ROOT)} ...")

    # Standard favicons (PNG variants for picky browsers + the .ico for legacy)
    render(MASTER, BRAND_DIR / "favicon-16.png", 16)
    render(MASTER, BRAND_DIR / "favicon-32.png", 32)
    render(MASTER, BRAND_DIR / "favicon-48.png", 48)
    render(MASTER, BRAND_DIR / "favicon-192.png", 192)
    render(MASTER, BRAND_DIR / "favicon-512.png", 512)

    # Apple touch icon (must be 180x180 to satisfy iOS home-screen)
    render(MASTER, BRAND_DIR / "apple-touch-icon.png", 180)

    # PWA maskable / regular icons
    render(MASTER, BRAND_DIR / "icon-192.png", 192)
    render(MASTER, BRAND_DIR / "icon-512.png", 512)

    # The multi-size .ico for IE / pinned-tab fallback
    build_ico()

    # OG / social preview
    print(f"Building OG image from {OG_MASTER.relative_to(REPO_ROOT)} ...")
    render(OG_MASTER, BRAND_DIR / "og-image.png", (1200, 630))

    print("done.")


if __name__ == "__main__":
    main()
