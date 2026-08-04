#!/usr/bin/env python3
"""Animate one trajectory as a GIF that shows *where* each action landed.

The older SOLAR trace GIFs print the action as text under the grid ("9 Color9"),
which leaves three things unreadable: which cells the action applied to, what it
changed, and how far along the episode is. Worse, they tint the selection in a
colour the ARC palette also uses, so a highlighted region is indistinguishable
from grid content.

This renders each action as two frames — the grid with its selection outlined
just before the action, then the result with the changed cells ringed — over a
timeline strip of every operation in the episode, with the worked examples and
the target kept on screen so the rule stays legible.

    python make_teaser_gif.py --root /hdd_data/yunho/ARC_best10_r3/whole \
        --task 05f2a901 --out figure/teaser.gif
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ARC_COLORS = ["#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00", "#AAAAAA",
              "#F012BE", "#FF851B", "#7FDBFF", "#870C25", "#F0F0F0"]
ARC_CMAP = mcolors.ListedColormap(ARC_COLORS)
ARC_NORM = mcolors.BoundaryNorm(boundaries=list(range(12)), ncolors=11)

# A marker must never be mistakable for a cell colour. Any hue collides with
# something in a 10-colour palette — cyan sat right on top of colour 8 and the
# "changed" ring vanished into the object it was marking. White ringed in black
# reads on all ten, and since selection and result never share a frame the two
# markers are told apart by line style, not hue.
MARK_EDGE = "#FFFFFF"
MARK_CASE = "#101010"


def draw_grid(ax, grid, h, w, box=None, sel=None, diff=None, title="",
              title_size=8, title_color="#222222"):
    arr = np.asarray(grid, dtype=int)[:h, :w]
    ax.imshow(arr, cmap=ARC_CMAP, norm=ARC_NORM, interpolation="nearest", zorder=1)
    lw = 0.5 if max(h, w) <= 16 else (0.35 if max(h, w) <= 24 else 0.25)
    for x in np.arange(-0.5, w, 1):
        ax.axvline(x, color="#FFFFFF", linewidth=lw, zorder=2)
    for y in np.arange(-0.5, h, 1):
        ax.axhline(y, color="#FFFFFF", linewidth=lw, zorder=2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.add_patch(plt.Rectangle((-0.5, -0.5), w, h, fill=False,
                               edgecolor="#909090", linewidth=0.8, zorder=6))

    def ring(mask, wid, style):
        """Outline the mask cell by cell — an outline survives any fill colour."""
        m = np.asarray(mask, dtype=bool)[:h, :w]
        for r, c in zip(*np.nonzero(m)):
            for col, lwid, z in ((MARK_CASE, wid + 1.2, 7), (MARK_EDGE, wid, 8)):
                ax.add_patch(mpatches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=col,
                    linewidth=lwid, linestyle=style, zorder=z))

    if sel is not None:
        ring(sel, 1.5, "solid")
    if diff is not None:
        ring(diff, 1.6, (0, (2.2, 1.6)))

    if title:
        ax.set_title(title, fontsize=title_size, pad=3, color=title_color)
    if box:
        BH, BW = box
        ax.set_xlim(-0.7, BW - 0.3)
        ax.set_ylim(BH - 0.3, -0.7)
    ax.set_aspect("equal")


def draw_timeline(ax, ops, cur, done):
    """One chip per operation: what has run, what is running, what is left."""
    ax.set_xlim(0, max(len(ops), 1)); ax.set_ylim(0, 1); ax.axis("off")
    for i, name in enumerate(ops):
        if i < done:
            fc, tc, ec = "#D8E6D2", "#33613f", "#B7CFAE"
        elif i == cur:
            fc, tc, ec = "#1F6F4A", "#FFFFFF", "#12432c"
        else:
            fc, tc, ec = "#F0F0F0", "#909090", "#DDDDDD"
        ax.add_patch(mpatches.FancyBboxPatch(
            (i + 0.06, 0.2), 0.88, 0.6, boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=fc, edgecolor=ec, linewidth=0.8))
        short = name.replace("FloodFill", "FF").replace("Rotate", "Rot")
        ax.text(i + 0.5, 0.5, short, ha="center", va="center",
                fontsize=min(7.5, 46 / max(len(ops), 1)), color=tc, weight="bold")


def frame(d, ops, step, phase, box, ex_n, task, dpi):
    """phase 0 = about to act (selection shown), 1 = acted (diff shown)."""
    grids, dims = d["grid"], d["grid_dim"]
    sel_masks = d["selection_mask"]
    before = np.array(grids[step]); hb, wb = dims[step]
    after = np.array(grids[step + 1]); ha, wa = dims[step + 1]

    if phase == 0:
        show, (h, w) = before, (hb, wb)
        sel = np.array(sel_masks[step]); diff = None
        head = f"{step + 1}. {ops[step]}   —   selection"
    else:
        show, (h, w) = after, (ha, wa)
        sel = None
        if (hb, wb) == (ha, wa):
            diff = (np.array(before)[:h, :w] != np.array(after)[:h, :w])
        else:
            diff = None
        head = f"{step + 1}. {ops[step]}   —   result"

    fig = plt.figure(figsize=(8.4, 4.5), dpi=dpi)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 2.5, 1.0],
                          height_ratios=[6.0, 1.0],
                          left=0.02, right=0.98, top=0.87, bottom=0.10,
                          wspace=0.12, hspace=0.28)

    # left: the worked examples that state the rule, centred against the canvas
    exg = gs[0, 0].subgridspec(3, 1, height_ratios=[1, 9, 1])[1].subgridspec(
        ex_n, 2, wspace=0.12, hspace=0.30)
    ex_box = (max(max(d["ex_in_grid_dim"][i][0], d["ex_out_grid_dim"][i][0])
                  for i in range(ex_n)),
              max(max(d["ex_in_grid_dim"][i][1], d["ex_out_grid_dim"][i][1])
                  for i in range(ex_n)))
    for i in range(ex_n):
        a1 = fig.add_subplot(exg[i, 0]); a2 = fig.add_subplot(exg[i, 1])
        eh, ew = d["ex_in_grid_dim"][i]
        draw_grid(a1, d["ex_in"][i], eh, ew, box=ex_box,
                  title="example in" if i == 0 else "", title_size=6.5)
        eh, ew = d["ex_out_grid_dim"][i]
        draw_grid(a2, d["ex_out"][i], eh, ew, box=ex_box,
                  title="out" if i == 0 else "", title_size=6.5)

    ax_main = fig.add_subplot(gs[0, 1])
    draw_grid(ax_main, show, h, w, box=box, sel=sel, diff=diff,
              title=head, title_size=11,
              title_color="#12432c" if phase else "#8a5a00")

    ax_t = fig.add_subplot(gs[0, 2])
    th, tw = d["grid_dim"][-1]
    draw_grid(ax_t, d["grid"][-1], th, tw, box=box, title="target", title_size=7)

    draw_timeline(fig.add_subplot(gs[1, :]), ops, step, step + phase)

    fig.suptitle(f"{task}   ·   step {step + 1} of {len(ops)}",
                 fontsize=12, y=0.965, color="#222222")
    # under the timeline, not beside the target — it collided with that title
    fig.text(0.5, 0.012, "solid ring = cells this action selects        "
             "dashed ring = cells the action changed",
             ha="center", fontsize=7.5, color="#777777")

    fig.canvas.draw()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                           fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1)
    plt.close(fig)
    return img.convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/hdd_data/yunho/ARC_best10_r3/whole")
    ap.add_argument("--task", required=True)
    ap.add_argument("--file", default=None)
    ap.add_argument("--out", default="figure/teaser.gif")
    ap.add_argument("--max_steps", type=int, default=12)
    ap.add_argument("--examples", type=int, default=2)
    ap.add_argument("--ms", type=int, default=900, help="ms per frame")
    ap.add_argument("--hold", type=int, default=2600, help="ms on the final frame")
    ap.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args()

    if args.file:
        path = Path(args.file)
    else:
        cands = sorted(Path(args.root).glob(f"test.{args.task}.*/*.json"))
        if not cands:
            raise SystemExit(f"no trajectory for {args.task} under {args.root}")
        path = cands[0]
    d = json.loads(path.read_text())
    ops = d["operation_name"]
    n = min(len(ops), args.max_steps)
    ex_n = min(args.examples, len(d["ex_in"]))

    dims = d["grid_dim"]
    box = (max(hw[0] for hw in dims), max(hw[1] for hw in dims))

    frames, durations = [], []
    for s in range(n):
        for phase in (0, 1):
            frames.append(frame(d, ops, s, phase, box, ex_n, args.task, args.dpi))
            durations.append(args.ms if phase == 0 else int(args.ms * 1.15))
    durations[-1] = args.hold

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    kb = out.stat().st_size / 1024
    print(f"{out}  {len(frames)} frames, {frames[0].size[0]}x{frames[0].size[1]}, "
          f"{kb:.0f} KB  ({' → '.join(ops[:n])}{' …' if n < len(ops) else ''})")


if __name__ == "__main__":
    main()
