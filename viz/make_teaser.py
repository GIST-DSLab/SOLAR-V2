#!/usr/bin/env python3
"""Render one trajectory as a static teaser figure: rule on top, actions below.

The viewer (`viz_trajectories.py`) is Streamlit, so it can show this but cannot
export it. This is the same palette and the same grid renderer, laid out as a
single figure for a README or a paper: the worked example pairs across the top
establish the rule, the strip below walks the policy's actions one at a time,
each panel showing the grid the operation is about to be applied to with its
selection highlighted.

Alignment note: record index i holds the state *before* operation i together
with the selection used for it, so panel i is honestly "what the agent saw and
what it chose". The last panel is the state after the final Submit.

    python make_teaser.py --list --min_steps 4 --max_steps 7
    python make_teaser.py --task 05269061 --out figure/teaser.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

DATA_ROOT = Path("/hdd_data/yunho")

ARC_COLORS = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00", "#AAAAAA",
    "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
    "#F0F0F0",  # 10 = padding / out of bounds
]
ARC_CMAP = mcolors.ListedColormap(ARC_COLORS)
ARC_NORM = mcolors.BoundaryNorm(boundaries=list(range(12)), ncolors=11)


def render_grid(ax, grid, h, w, title="", sel_mask=None, highlight="#00E5FF",
                title_size=8, title_color="black", box=None):
    """Draw one grid. `box` = (H, W) pins the axes extent so every panel in a
    strip shares one cell size — a ResizeGrid then reads as the canvas growing
    instead of as an unrelated picture at a different zoom."""
    arr = np.asarray(grid, dtype=int)[:h, :w]
    ax.imshow(arr, cmap=ARC_CMAP, norm=ARC_NORM, interpolation="nearest",
              zorder=1)
    lw = 0.4 if max(h, w) <= 20 else 0.25
    for x in np.arange(-0.5, w, 1):
        ax.axvline(x, color="white", linewidth=lw, zorder=3)
    for y in np.arange(-0.5, h, 1):
        ax.axhline(y, color="white", linewidth=lw, zorder=3)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    # outline the grid itself, since the axes frame no longer marks its edge
    ax.add_patch(plt.Rectangle((-0.5, -0.5), w, h, fill=False,
                               edgecolor="#888888", linewidth=0.7, zorder=4))
    if title:
        ax.set_title(title, fontsize=title_size, pad=4, color=title_color)

    if sel_mask is not None:
        mask = np.asarray(sel_mask, dtype=bool)[:h, :w]
        if mask.any():
            # A full-grid selection tinted every panel the same colour and hid
            # the actual content, so fill only sparse selections and always
            # trace the boundary, which reads at any coverage.
            if mask.mean() < 0.5:
                overlay = np.zeros((*mask.shape, 4))
                overlay[mask] = mcolors.to_rgba(highlight, alpha=0.30)
                ax.imshow(overlay, interpolation="nearest", zorder=2)
            padded = np.pad(mask.astype(float), 1)
            ax.contour(padded, levels=[0.5], colors=[highlight], linewidths=1.6,
                       extent=(-1.5, w + 0.5, h + 0.5, -1.5), zorder=5)

    if box:
        BH, BW = box
        ax.set_xlim(-0.6, BW - 0.4)
        ax.set_ylim(BH - 0.4, -0.6)
    ax.set_aspect("equal")


def find_files(root: Path, task: str | None):
    out = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        parts = folder.name.split(".")
        if len(parts) < 2:
            continue
        tid = parts[1]
        if task and tid != task:
            continue
        out.extend((tid, f) for f in sorted(folder.glob("*.json")))
    return out


def teaser(d: dict, task: str, out_path: Path, max_examples: int = 3,
           max_steps: int | None = None, title: str | None = None) -> None:
    ops = d["operation_name"]
    n = len(ops)
    shown = n if max_steps is None else min(n, max_steps)
    truncated = shown < n

    ex_in, ex_out = d["ex_in"], d["ex_out"]
    ex_ind, ex_outd = d["ex_in_grid_dim"], d["ex_out_grid_dim"]
    k = min(max_examples, len(ex_in))

    # +1 for the final state panel, +1 more when we elide the middle
    n_panels = shown + 1 + (1 if truncated else 0)
    ncols = max(2 * k, n_panels)

    # one cell size per strip, so panels are comparable within a row
    ex_box = (max(max(ex_ind[i][0], ex_outd[i][0]) for i in range(k)),
              max(max(ex_ind[i][1], ex_outd[i][1]) for i in range(k)))
    dims = d["grid_dim"][:shown] + [d["grid_dim"][-1]]
    tr_box = (max(hw[0] for hw in dims), max(hw[1] for hw in dims))

    # Each panel is `cell` wide, so its height follows its box aspect. Deriving
    # the figure height from that keeps tall grids from leaving a band of dead
    # space under short ones.
    cell = 1.35
    TITLE_IN = 0.42
    top_h = cell * ex_box[0] / ex_box[1] + TITLE_IN
    bot_h = cell * tr_box[0] / tr_box[1] + TITLE_IN
    fig = plt.figure(figsize=(cell * ncols, top_h + bot_h), dpi=200)
    top, bottom = fig.subfigures(2, 1, height_ratios=[top_h, bot_h])

    top.suptitle(
        title or f"task {task} — the rule, from worked examples",
        fontsize=11, y=1 - 0.12 * TITLE_IN / top_h,
    )
    ax_top = top.subplots(1, ncols, squeeze=False)[0]
    for i in range(k):
        h, w = ex_ind[i]
        render_grid(ax_top[2 * i], ex_in[i], h, w, f"example {i+1}  in",
                    box=ex_box, title_size=7.5)
        h, w = ex_outd[i]
        render_grid(ax_top[2 * i + 1], ex_out[i], h, w, "out", box=ex_box,
                    title_size=7.5)
    for j in range(2 * k, ncols):
        ax_top[j].axis("off")

    bottom.suptitle(
        f"the trajectory — {n} ARCLE actions, one panel per action; "
        f"cyan marks the selection the action is applied to",
        fontsize=11, y=1 - 0.12 * TITLE_IN / bot_h,
    )
    axes = bottom.subplots(1, ncols, squeeze=False)[0]
    col = 0
    for i in range(shown):
        h, w = d["grid_dim"][i]
        render_grid(axes[col], d["grid"][i], h, w,
                    title=f"{i+1}. {ops[i]}", sel_mask=d["selection_mask"][i],
                    title_color="#0b6623", box=tr_box, title_size=7.5)
        col += 1
    if truncated:
        axes[col].text(0.5, 0.5, f"…\n+{n - shown}\nactions", ha="center",
                       va="center", fontsize=9, color="#555555")
        axes[col].axis("off")
        col += 1
    h, w = d["grid_dim"][-1]
    render_grid(axes[col], d["grid"][-1], h, w, title="output",
                title_color="#8b0000", box=tr_box, title_size=7.5)
    for j in range(col + 1, ncols):
        axes[j].axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"{out_path}  ({n} ops: {' → '.join(ops[:shown])}"
          f"{' …' if truncated else ''})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DATA_ROOT / "ARC_best10" / "whole"))
    ap.add_argument("--task", default=None)
    ap.add_argument("--file", default=None, help="explicit trajectory json")
    ap.add_argument("--out", default="figure/teaser.png")
    ap.add_argument("--max_steps", type=int, default=8)
    ap.add_argument("--max_examples", type=int, default=3)
    ap.add_argument("--title", default=None)
    ap.add_argument("--list", action="store_true",
                    help="print candidate tasks by op count instead of rendering")
    ap.add_argument("--min_steps", type=int, default=3)
    args = ap.parse_args()

    if args.list:
        root = Path(args.root)
        seen = {}
        for tid, f in find_files(root, args.task):
            if tid in seen:
                continue
            d = json.loads(f.read_text())
            n = len(d["operation_name"])
            if args.min_steps <= n <= args.max_steps:
                seen[tid] = (n, d["operation_name"], f)
        for tid, (n, names, f) in sorted(seen.items(), key=lambda x: x[1][0]):
            print(f"{tid}  {n:2d} ops  {' '.join(names)}")
        print(f"\n{len(seen)} tasks with {args.min_steps}-{args.max_steps} ops")
        return

    if args.file:
        path = Path(args.file)
        task = args.task or path.parent.name.split(".")[1]
    else:
        cands = find_files(Path(args.root), args.task)
        if not cands:
            raise SystemExit(f"no trajectory found for task={args.task}")
        task, path = cands[0]

    teaser(json.loads(path.read_text()), task, Path(args.out),
           max_examples=args.max_examples, max_steps=args.max_steps,
           title=args.title)


if __name__ == "__main__":
    main()
