#!/usr/bin/env python3
"""Render one recorded trajectory as a static figure.

The layout follows the original SOLAR-Generator visualiser: one row per
demonstration pair, input beside output, then the trajectory across the bottom
row — the test input first, then the grid at each step, each panel labelled
underneath with the ARCLE op number and name applied to it.

Two things are drawn differently. The selection an operation applies to is
outlined on its panel, so the row reads as a policy's choices rather than as a
sequence of snapshots; and grids are clipped to their true extent from
`grid_dim`, so the padding a 30x30 recording carries never reaches the page.

Alignment note: record index i holds the state *before* operation i together
with the selection used for it, so panel i is honestly "what the agent saw and
what it chose". The last panel is the state after the final Submit.

    python viz/visualize_trajectory.py --root <data_folder>/whole --list
    python viz/visualize_trajectory.py --root <data_folder>/whole --task 05269061
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


def figure(d: dict, task: str, out_path: Path, max_examples: int = 3,
           max_steps: int | None = None, title: str | None = None) -> None:
    ops, op_nums = d["operation_name"], d["operation"]
    n = len(ops)
    shown = n if max_steps is None else min(n, max_steps)
    truncated = shown < n

    ex_in, ex_out = d["ex_in"], d["ex_out"]
    ex_ind, ex_outd = d["ex_in_grid_dim"], d["ex_out_grid_dim"]
    k = min(max_examples, len(ex_in))

    # +1 for the final state panel, +1 more when we elide the middle
    n_panels = shown + 1 + (1 if truncated else 0)
    ncols = max(2, n_panels)

    # one cell size per row, so panels within a row stay comparable — a
    # ResizeGrid then reads as the canvas growing, not as a different zoom
    ex_box = (max(max(ex_ind[i][0], ex_outd[i][0]) for i in range(k)),
              max(max(ex_ind[i][1], ex_outd[i][1]) for i in range(k)))
    dims = d["grid_dim"][:shown] + [d["grid_dim"][-1]]
    tr_box = (max(hw[0] for hw in dims), max(hw[1] for hw in dims))

    # Panel width is fixed; height follows the box aspect, which is what keeps a
    # row of tall grids from leaving a band of dead space under a row of short
    # ones. The op label under each trajectory panel needs its own strip.
    # A row is `ncols` panels wide whatever it uses, so a demonstration row and
    # the trajectory row draw at one cell size. Row height is derived from the
    # width a panel actually gets once margins and wspace are taken out —
    # deriving it from `cell` alone left a band of white under every row.
    cell, WSPACE, SIDE = 1.35, 0.18, 0.99
    EX_TOP, EX_BOT = 0.80, 0.02          # axes band inside a demonstration row
    TR_TOP, TR_BOT = 0.82, 0.14          # the trajectory row keeps its op labels
    panel_w = cell * ncols * SIDE / (ncols + (ncols - 1) * WSPACE)
    ex_h = panel_w * ex_box[0] / ex_box[1] / (EX_TOP - EX_BOT)
    tr_h = panel_w * tr_box[0] / tr_box[1] / (TR_TOP - TR_BOT)
    # the first band is empty and holds the suptitle: drawn over the rows it
    # landed on the first demonstration's own label
    HEAD = 0.5
    fig = plt.figure(figsize=(cell * ncols, HEAD + ex_h * k + tr_h), dpi=200)
    bands = fig.subfigures(k + 2, 1, height_ratios=[HEAD] + [ex_h] * k + [tr_h])
    rows = bands[1:]

    fig.suptitle(title or f"task {task} — {n} ARCLE actions", fontsize=11)

    # demonstration pairs: input beside output, one pair per row
    for i in range(k):
        ax = rows[i].subplots(1, ncols, squeeze=False)[0]
        # the label breaks over two lines: at one panel wide it does not fit on
        # one, and a single line ran into the next panel's title
        h, w = ex_ind[i]
        render_grid(ax[0], ex_in[i], h, w, f"demonstration\ninput {i+1}",
                    box=ex_box, title_size=7.5)
        h, w = ex_outd[i]
        render_grid(ax[1], ex_out[i], h, w, f"demonstration\noutput {i+1}",
                    box=ex_box, title_size=7.5)
        for j in range(2, ncols):
            ax[j].axis("off")
        rows[i].subplots_adjust(left=(1 - SIDE) / 2, right=(1 + SIDE) / 2,
                                top=EX_TOP, bottom=EX_BOT, wspace=WSPACE)

    # the trajectory, one panel per step
    axes = rows[k].subplots(1, ncols, squeeze=False)[0]
    col = 0
    for i in range(shown):
        h, w = d["grid_dim"][i]
        render_grid(axes[col], d["grid"][i], h, w,
                    title="test input" if i == 0 else f"step {i}",
                    sel_mask=d["selection_mask"][i], box=tr_box, title_size=7.5)
        axes[col].text(0.5, -0.06, f"{op_nums[i]}  {ops[i]}", ha="center",
                       va="top", transform=axes[col].transAxes, fontsize=7,
                       color="#0b6623")
        col += 1
    if truncated:
        axes[col].text(0.5, 0.5, f"…\n+{n - shown}\nactions", ha="center",
                       va="center", fontsize=9, color="#555555")
        axes[col].axis("off")
        col += 1
    h, w = d["grid_dim"][-1]
    # `n`, not `shown`: with a truncated middle the last panel is still the
    # state after every action, and labelling it with the truncated count lied
    render_grid(axes[col], d["grid"][-1], h, w, title=f"step {n}",
                box=tr_box, title_size=7.5)
    axes[col].text(0.5, -0.06, "final grid", ha="center", va="top",
                   transform=axes[col].transAxes, fontsize=7, color="#8b0000")
    for j in range(col + 1, ncols):
        axes[j].axis("off")
    rows[k].subplots_adjust(left=(1 - SIDE) / 2, right=(1 + SIDE) / 2,
                            top=TR_TOP, bottom=TR_BOT, wspace=WSPACE)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"{out_path}  ({n} ops: {' → '.join(ops[:shown])}"
          f"{' …' if truncated else ''})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="a rollout directory, e.g. <data_folder>/whole")
    ap.add_argument("--task", default=None)
    ap.add_argument("--file", default=None, help="explicit trajectory json")
    ap.add_argument("--out", default="figure/trajectory.png")
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

    figure(json.loads(path.read_text()), task, Path(args.out),
           max_examples=args.max_examples, max_steps=args.max_steps,
           title=args.title)


if __name__ == "__main__":
    main()
