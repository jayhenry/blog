#!/usr/bin/env python3
"""Convert SVG files to JPG images.

Dependencies:
    python -m pip install cairosvg pillow

Usage:
    python articles/202606-cute-dsl/convert_svg2jpg.py
    python articles/202606-cute-dsl/convert_svg2jpg.py path/to/a.svg path/to/b.svg
    python articles/202606-cute-dsl/convert_svg2jpg.py articles/202606-cute-dsl/img --remove-source
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import cairosvg
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PATHS = [SCRIPT_DIR / "img"]


def convert_svg_to_jpg(svg_path: Path, quality: int) -> Path:
    jpg_path = svg_path.with_suffix(".jpg")

    # SVG 先渲染到 PNG，再转 RGB JPG，避免透明背景在 JPG 中变黑。
    png_buffer = BytesIO()
    cairosvg.svg2png(
        url=str(svg_path),
        write_to=png_buffer,
        background_color="white",
    )
    png_buffer.seek(0)

    with Image.open(png_buffer) as image:
        image.convert("RGB").save(
            jpg_path,
            "JPEG",
            quality=quality,
            subsampling=0,
            optimize=True,
        )

    return jpg_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="*",
        type=Path,
        help="SVG files or directories. Defaults to this article directory's img folder.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality, from 1 to 95. Default: 95.",
    )
    parser.add_argument(
        "--remove-source",
        action="store_true",
        help="Remove each SVG after the JPG is written.",
    )
    return parser.parse_args()


def find_svg_paths(paths: list[Path]) -> list[Path]:
    svg_paths: list[Path] = []
    for path in paths:
        if path.is_dir():
            svg_paths.extend(sorted(path.glob("*.svg")))
        else:
            svg_paths.append(path)
    return svg_paths


def main() -> None:
    args = parse_args()
    svg_paths = find_svg_paths(args.path or DEFAULT_PATHS)

    for svg_path in svg_paths:
        if not svg_path.exists():
            raise FileNotFoundError(svg_path)
        jpg_path = convert_svg_to_jpg(svg_path, args.quality)
        if args.remove_source:
            svg_path.unlink()
        print(jpg_path)


if __name__ == "__main__":
    main()
