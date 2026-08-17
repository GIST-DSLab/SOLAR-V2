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

    python make_teaser_gif.py --root <data_folder>/whole \
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
from matplotlib.collections import LineCollection
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

    def outline(mask, wid, style):
        """Trace the region's border once, along the actual cell edges.

        Ringing every selected cell drew a hatch over any multi-cell selection —
        exact, but the busiest thing on screen. `contour` calms it down but is
        wrong twice over: it samples at grid points rather than cell centres, so
        the outline landed two rows off, and it interpolates, so square corners
        came out rounded. Walking the mask and emitting the edges that separate
        it from everything else is exact and crisp.
        """
        m = np.asarray(mask, dtype=bool)[:h, :w]
        if not m.any():
            return
        segs = []
        for r, c in zip(*np.nonzero(m)):
            if r == 0 or not m[r - 1, c]:
                segs.append([(c - 0.5, r - 0.5), (c + 0.5, r - 0.5)])
            if r == h - 1 or not m[r + 1, c]:
                segs.append([(c - 0.5, r + 0.5), (c + 0.5, r + 0.5)])
            if c == 0 or not m[r, c - 1]:
                segs.append([(c - 0.5, r - 0.5), (c - 0.5, r + 0.5)])
            if c == w - 1 or not m[r, c + 1]:
                segs.append([(c + 0.5, r - 0.5), (c + 0.5, r + 0.5)])
        for col, lwid, z in ((MARK_CASE, wid + 2.4, 7), (MARK_EDGE, wid, 8)):
            ax.add_collection(LineCollection(
                segs, colors=col, linewidths=lwid, linestyles=style,
                capstyle="round", zorder=z))

    if sel is not None:
        outline(sel, 3.2, "solid")
    if diff is not None:
        outline(diff, 3.0, "dashed")

    if title:
        ax.set_title(title, fontsize=title_size, pad=3, color=title_color)
    if box:
        BH, BW = box
        ax.set_xlim(-0.7, BW - 0.3)
        ax.set_ylim(BH - 0.3, -0.7)
    ax.set_aspect("equal")


def runs_of(ops):
    """Collapse consecutive identical ops into (name, count, first_index)."""
    out = []
    for i, name in enumerate(ops):
        if out and out[-1][0] == name:
            out[-1][1] += 1
        else:
            out.append([name, 1, i])
    return [tuple(r) for r in out]


def draw_buttons(ax, ops, step, pressed):
    """The action space as buttons, the current one held down.

    One chip per step turned MoveL x7 into seven identical chips, which read as
    seven different things happening. Collapsing a run to a single button and
    pressing it once per step matches what the episode is actually doing, and
    the press is what makes it read as an action being taken rather than a label
    being highlighted.
    """
    rs = runs_of(ops)
    ax.set_xlim(0, max(len(rs), 1)); ax.set_ylim(0, 1); ax.axis("off")
    fs = float(np.clip(46 / max(len(rs), 1), 5.5, 9.0))

    for j, (name, count, first) in enumerate(rs):
        last = first + count - 1
        is_cur = first <= step <= last
        if step > last:
            face, tc = "#DCE8D6", "#5E7A62"
        elif is_cur:
            # Colour alone carries the press: dark while the action is being
            # issued, lighter once it has been. A face that physically travelled
            # moved a couple of pixels at this size and read as nothing.
            face, tc = ("#1F6F4A", "#FFFFFF") if pressed else ("#7FB295", "#FFFFFF")
        else:
            face, tc = "#F4F4F4", "#BEBEBE"

        ax.add_patch(mpatches.FancyBboxPatch(
            (j + 0.08, 0.30), 0.84, 0.40,
            boxstyle="round,pad=0.015,rounding_size=0.10",
            facecolor=face, edgecolor="none"))
        short = name.replace("FloodFill", "FF").replace("Rotate", "Rot")
        ax.text(j + 0.5, 0.50, short, ha="center", va="center",
                fontsize=fs, color=tc, weight="bold")


def _unused_draw_sequence(fig, ops, cur, y=0.055):
    """Kept for reference: the same information as one line of text."""
    short = [n.replace("FloodFill", "FF").replace("Rotate", "Rot") for n in ops]
    size = float(np.clip(66 / max(len(short), 1), 6.0, 10.5))
    sep = "   ·   "
    # Advance by what was actually rendered. Estimating token widths from character
    # counts stacked every label on top of the last one.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    W = fig.get_size_inches()[0] * fig.dpi

    def width_of(s, weight):
        t = fig.text(0, -1, s, fontsize=size, weight=weight)
        w = t.get_window_extent(renderer=rend).width / W
        t.remove()
        return w

    def lay_out():
        out = []
        for i, s in enumerate(short):
            weight = "bold" if i == cur else "normal"
            out.append((s, weight, width_of(s, weight)))
            if i < len(short) - 1:
                out.append((sep, "normal", width_of(sep, "normal")))
        return out

    # A nine-op episode ran off both edges of the frame at the nominal size, so
    # shrink to fit rather than clip — the sequence is only useful entire.
    tokens = lay_out()
    total = sum(t[2] for t in tokens)
    if total > 0.94:
        size = max(4.6, size * 0.94 / total)
        tokens = lay_out()

    x = 0.5 - sum(t[2] for t in tokens) / 2
    idx = 0
    for s, weight, wid in tokens:
        if s == sep:
            colour = "#D0D0D0"
        else:
            colour = ("#12432c" if idx == cur
                      else ("#8FA894" if idx < cur else "#C4C4C4"))
            idx += 1
        fig.text(x, y, s, fontsize=size, ha="left", va="center",
                 color=colour, weight=weight)
        x += wid


def frame(d, ops, step, phase, box, ex_n, task, dpi):
    """phase 0 = about to act (selection shown), 1 = acted (diff shown)."""
    grids, dims = d["grid"], d["grid_dim"]
    sel_masks = d["selection_mask"]
    before = np.array(grids[step]); hb, wb = dims[step]
    after = np.array(grids[step + 1]); ha, wa = dims[step + 1]

    if phase == 0:
        show, (h, w) = before, (hb, wb)
        sel = np.array(sel_masks[step]); diff = None
        head = ops[step]
    else:
        show, (h, w) = after, (ha, wa)
        sel = None
        # For an object op the environment keeps the object selected, so
        # `selected` after the step is the object at its new place. Marking the
        # changed cells instead outlined the vacated square as well, which read
        # as the object having been left behind. Fall back to the diff only for
        # ops that hold no object.
        moved = np.array(d["selected"][step + 1])[:h, :w].astype(bool)
        if moved.any():
            diff = moved
        elif (hb, wb) == (ha, wa):
            diff = (np.array(before)[:h, :w] != np.array(after)[:h, :w])
        else:
            diff = None
        # The op name stays put across both frames while the small line above it
        # changes, so the pair reads as one decision — predicted, then carried
        # out — rather than as two unrelated captions.
        head = ops[step]

    BH, BW = box
    main_w = float(np.clip(3.0 * (BW / BH), 1.4, 4.2))
    ex_w = main_w * 0.86
    tgt_w = main_w * 0.34
    fig_w = float(np.clip(0.5 + (ex_w + main_w + tgt_w) * 1.05, 4.6, 8.6))
    fig = plt.figure(figsize=(fig_w, 4.0), dpi=dpi)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 3, width_ratios=[ex_w, main_w, tgt_w],
                          height_ratios=[9.0, 1.0],
                          left=0.02, right=0.98, top=0.80, bottom=0.05,
                          wspace=0.14, hspace=0.22)

    # The worked example pairs, back on the left: with the sequence strip and the
    # legend gone there is room for them, and without them a viewer has no way to
    # tell what the episode is trying to do.
    exg = gs[0, 0].subgridspec(3, 1, height_ratios=[0.4, 12, 0.4])[1].subgridspec(
        ex_n, 2, wspace=0.08, hspace=0.26)
    ex_box = (max(max(d["ex_in_grid_dim"][i][0], d["ex_out_grid_dim"][i][0])
                  for i in range(ex_n)),
              max(max(d["ex_in_grid_dim"][i][1], d["ex_out_grid_dim"][i][1])
                  for i in range(ex_n)))
    for i in range(ex_n):
        eh, ew = d["ex_in_grid_dim"][i]
        draw_grid(fig.add_subplot(exg[i, 0]), d["ex_in"][i], eh, ew, box=ex_box,
                  title=f"demonstration input {i + 1}", title_size=7.5)
        eh, ew = d["ex_out_grid_dim"][i]
        draw_grid(fig.add_subplot(exg[i, 1]), d["ex_out"][i], eh, ew, box=ex_box,
                  title=f"demonstration output {i + 1}", title_size=7.5)

    ax_main = fig.add_subplot(gs[0, 1])
    draw_grid(ax_main, show, h, w, box=box, sel=sel, diff=diff)
    ax_main.text(0.5, 1.018, head, transform=ax_main.transAxes, ha="center",
                 va="bottom", fontsize=16, weight="bold",
                 color="#12432c" if phase else "#B26B00")

    ax_t = fig.add_subplot(gs[0, 2])
    th, tw = d["grid_dim"][-1]
    draw_grid(ax_t, d["grid"][-1], th, tw, box=box, title="target", title_size=8)

    # phase 0 = the button held down, phase 1 = released with the action done
    draw_buttons(fig.add_subplot(gs[1, :]), ops, step, pressed=(phase == 0))

    fig.suptitle(f"ARC task {task}", fontsize=11, y=0.945, color="#555555")

    fig.canvas.draw()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                           fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1)
    plt.close(fig)
    return img.convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="a rollout directory, e.g. <data_folder>/whole")
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
