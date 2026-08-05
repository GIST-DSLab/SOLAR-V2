#!/usr/bin/env python3
"""Build the overview page's backdrop: a mosaic of real ARC grids.

The teaser figure is a labelled explainer on white — as a background it reads as
noise behind the copy. This tiles the `preview` config's rendered grids into a
wide masonry on near-black, which is what the hero actually wants: enough colour
and structure to say "ARC", nothing legible enough to compete with the text.

Deliberately *not* blurred here. Flat colour blocks quantise to a tiny PNG-8;
blurring them first replaces every flat region with a gradient and multiplies
the file size. The page applies the blur in CSS, where it is free to tune.

    python viz/build_hero_bg.py --preview /hdd_data/yunho/release_preview/preview.parquet \
                                --out docs/hero.png
"""
from __future__ import annotations

import argparse
import io
import random
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

GROUND = (16, 16, 18)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", default="/hdd_data/yunho/release_preview/preview.parquet")
    ap.add_argument("--out", default="docs/hero.png")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--col", type=int, default=176, help="masonry column width")
    ap.add_argument("--gap", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    table = pq.read_table(args.preview)
    raw = []
    for col in ("input", "target"):
        for cell in table.column(col):
            raw.append(cell.as_py()["bytes"])
    rng = random.Random(args.seed)
    rng.shuffle(raw)

    # This draw leans heavily on one background colour, and a mosaic of ~60 grids
    # picked at random comes out almost entirely magenta. Bucket by dominant
    # colour and deal round-robin: still real grids, spread across the palette.
    buckets: dict[tuple, list] = {}
    for b in raw[:400]:
        im = Image.open(io.BytesIO(b)).convert("RGB")
        dom = max(im.getcolors(maxcolors=1 << 20))[1]
        buckets.setdefault(dom, []).append(b)
    order = sorted(buckets, key=lambda k: -len(buckets[k]))
    pool = []
    while any(buckets[k] for k in order):
        for k in order:
            if buckets[k]:
                pool.append(buckets[k].pop())

    canvas = Image.new("RGB", (args.width, args.height), GROUND)
    ncol = (args.width + args.gap) // (args.col + args.gap)
    # centre the columns rather than leaving the remainder on one edge
    x0 = (args.width - (ncol * args.col + (ncol - 1) * args.gap)) // 2

    i = 0
    for c in range(ncol):
        x = x0 + c * (args.col + args.gap)
        # stagger the start so the top edge is not a straight line of grids
        y = -random.Random(args.seed + c).randrange(40, 200)
        while y < args.height:
            im = Image.open(io.BytesIO(pool[i % len(pool)]))
            i += 1
            h = max(1, round(im.height * args.col / im.width))
            if h > args.col * 2.2:            # very tall grids dominate a column
                continue
            canvas.paste(im.resize((args.col, h), Image.NEAREST), (x, y))
            y += h + args.gap

    out = Path(args.out)
    # 10 ARC colours plus the ground: an 8-bit palette is exact here, not lossy
    canvas.convert("P", palette=Image.ADAPTIVE, colors=32).save(out, optimize=True)
    print(f"{out}  {canvas.size[0]}x{canvas.size[1]}  {out.stat().st_size/1e3:.0f} kB")


if __name__ == "__main__":
    main()
