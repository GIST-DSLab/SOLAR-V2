#!/usr/bin/env python3
"""Build a `preview` config the Hub's dataset viewer can actually show.

The shipped configs pack every array as a raw byte blob, which is what makes
4,000 episodes fit in 4 MB — and what makes the viewer useless on them: 23 of
the 54 columns render as unreadable binary. This writes a second, small config
whose columns are pictures, so opening the dataset shows grids instead of bytes.

One row per task: the input, the target, and a filmstrip of the whole episode
with each action's selection outlined. Rendering is plain PIL — matplotlib would
be ~20x slower over 400 tasks and is not needed for flat colour cells.

    python build_preview.py --out /hdd_data/yunho/release_preview
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

ARC_RGB = np.array([
    (0x00, 0x00, 0x00), (0x00, 0x74, 0xD9), (0xFF, 0x41, 0x36), (0x2E, 0xCC, 0x40),
    (0xFF, 0xDC, 0x00), (0xAA, 0xAA, 0xAA), (0xF0, 0x12, 0xBE), (0xFF, 0x85, 0x1B),
    (0x7F, 0xDB, 0xFF), (0x87, 0x0C, 0x25), (0xF0, 0xF0, 0xF0),
], dtype=np.uint8)

CELL = 14          # pixels per grid cell
SEP = (255, 255, 255)
PAD = (255, 255, 255)


def render(grid, h, w, sel=None, cell=CELL) -> Image.Image:
    """One grid as an RGB image, with an optional selection outlined in white."""
    g = np.asarray(grid, dtype=int)[:h, :w]
    img = ARC_RGB[np.clip(g, 0, 10)]
    img = np.kron(img, np.ones((cell, cell, 1), dtype=np.uint8))
    # 1px separators, so cells stay countable
    img[::cell, :] = SEP
    img[:, ::cell] = SEP

    if sel is not None:
        # Only the region's border. Outlining each selected cell turns any
        # multi-cell selection into a hatch, and Submit selects the whole grid,
        # which came out as a white mesh over the entire picture.
        m = np.asarray(sel, dtype=bool)[:h, :w]
        for r, c in zip(*np.nonzero(m)):
            y0, x0 = r * cell, c * cell
            y1, x1 = y0 + cell, x0 + cell
            for t, col in ((0, (16, 16, 16)), (1, (255, 255, 255)), (2, (255, 255, 255))):
                if r == 0 or not m[r - 1, c]:
                    img[y0 + t, x0:x1] = col
                if r == h - 1 or not m[r + 1, c]:
                    img[y1 - 1 - t, x0:x1] = col
                if c == 0 or not m[r, c - 1]:
                    img[y0:y1, x0 + t] = col
                if c == w - 1 or not m[r, c + 1]:
                    img[y0:y1, x1 - 1 - t] = col
    return Image.fromarray(img)


def filmstrip(d, cell=10, gap=10, max_steps=12) -> Image.Image:
    """Every state of the episode in a row, each action's selection outlined."""
    ops = d["operation_name"]
    n = min(len(ops), max_steps)
    frames = []
    for i in range(n):
        h, w = d["grid_dim"][i]
        frames.append(render(d["grid"][i], h, w, d["selection_mask"][i], cell))
    h, w = d["grid_dim"][-1]
    frames.append(render(d["grid"][-1], h, w, None, cell))

    W = sum(f.width for f in frames) + gap * (len(frames) - 1)
    H = max(f.height for f in frames)
    strip = Image.new("RGB", (W, H), PAD)
    x = 0
    for f in frames:
        strip.paste(f, (x, (H - f.height) // 2))
        x += f.width + gap
    return strip


def png(img: Image.Image) -> dict:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return {"bytes": buf.getvalue(), "path": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/hdd_data/yunho/ARC_best10_r4/whole")
    ap.add_argument("--out", default="/hdd_data/yunho/release_preview")
    ap.add_argument("--max_steps", type=int, default=12)
    args = ap.parse_args()

    from datasets import Dataset, Features, Image as HFImage, Value

    rows = []
    for folder in sorted(Path(args.root).iterdir()):
        if not folder.is_dir():
            continue
        task = folder.name.split(".")[1]
        f = sorted(folder.glob("*.json"))[0]
        d = json.loads(f.read_text())
        ops = d["operation_name"]
        h0, w0 = d["grid_dim"][0]
        hN, wN = d["grid_dim"][-1]
        rows.append({
            "task_id": task,
            "n_actions": len(ops),
            "operations": " → ".join(ops),
            "input": png(render(d["grid"][0], h0, w0)),
            "target": png(render(d["grid"][-1], hN, wN)),
            "trajectory": png(filmstrip(d, max_steps=args.max_steps)),
        })
        if len(rows) % 100 == 0:
            print(f"  {len(rows)} tasks", flush=True)

    feats = Features({
        "task_id": Value("string"),
        "n_actions": Value("int32"),
        "operations": Value("string"),
        "input": HFImage(),
        "target": HFImage(),
        "trajectory": HFImage(),
    })
    ds = Dataset.from_list(rows, features=feats)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "preview.parquet"
    ds.to_parquet(str(path))
    print(f"{len(ds)} rows -> {path}  ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
