"""
ARC Task: 1a07d186 (RE-ARC) — LLM-generated grid_maker
"""
from __future__ import annotations

import inspect

import sys
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

SOLAR_ROOT = Path(__file__).resolve().parents[3]
if str(SOLAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLAR_ROOT))

REARC_ROOT = SOLAR_ROOT / "re-arc"
_rs = str(REARC_ROOT)
while _rs in sys.path:
    sys.path.remove(_rs)
sys.path.insert(0, _rs)

from maker.base_grid_maker import BaseGridMaker

import importlib
for _m in ["utils", "dsl", "generators"]:
    if _m in sys.modules:
        del sys.modules[_m]
from utils import *  # noqa: F401,F403  (unifint, choice, sample, etc.)
from dsl import *    # noqa: F401,F403

# ── LLM-generated: sample_colors / generate / derive_operations ───────────────
import random
import numpy as np
from collections import Counter

from maker.sel_helpers import sel_of


# --------------------------------------------------------------------------
# 1. sample_colors
# --------------------------------------------------------------------------
# The generator samples: bgc, the line colours, and the noise colours.
# We fix bgc for the whole episode and fix a colour *ordering* (palette) so the
# line-colour roles stay stable across every instance of the episode.
# The only discrete structural variant is the final dmirror (lines are columns
# vs. lines are rows) -> planned per instance so both cases are shown.

VARIANTS = [{"mirror": False}, {"mirror": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    palette = [c for c in cols if c != bgc]
    random.shuffle(palette)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]

    return {"bgc": bgc, "palette": palette, "instance_plan": plan}


# --------------------------------------------------------------------------
# 2. generate
# --------------------------------------------------------------------------
def generate(diff_lb, diff_ub, max_h, max_w, bgc, palette=None, mirror=None) -> dict:
    cols = list(range(10))
    if palette is None:
        palette = [c for c in cols if c != bgc]
        random.shuffle(palette)
    if mirror is None:
        mirror = random.choice([True, False])

    hub = max(8, min(30, int(max_h)))
    wub = max(8, min(30, int(max_w)))
    h = unifint(diff_lb, diff_ub, (8, hub))
    w = unifint(diff_lb, diff_ub, (8, wub))

    nlines = unifint(diff_lb, diff_ub, (1, max(1, w // 5)))
    linecols = list(palette[:nlines])
    remcols = list(palette[nlines:])
    nnoisecols = unifint(diff_lb, diff_ub, (0, len(remcols)))
    noisecols = list(random.sample(remcols, nnoisecols)) if nnoisecols > 0 else []

    locopts = list(range(w))
    locs = []
    for _ in range(nlines):
        if len(locopts) == 0:
            break
        loc = random.choice(locopts)
        locopts = [x for x in locopts if not (loc - 2 <= x <= loc + 2)]
        locs.append(loc)
    locs = sorted(locs)
    nlines = len(locs)
    linecols = linecols[:nlines]

    gi = canvas(bgc, (h, w))
    for loc, col in zip(locs, linecols):
        gi = fill(gi, col, connect((0, loc), (h - 1, loc)))
    go = tuple(e for e in gi)

    nilocs = unifint(diff_lb, diff_ub, (1, h))
    ilocs = random.sample(list(range(h)), nilocs)
    dotlocopts = [x for x in range(w) if x not in locs]
    allcols = sorted(set(linecols) | set(noisecols))

    for ii in ilocs:
        ub = min(nlines + nnoisecols, (w - nlines) // 2 - 1)
        ub = max(1, ub)
        ndots = unifint(diff_lb, diff_ub, (1, ub))
        ndots = min(ndots, len(dotlocopts), len(allcols))
        if ndots <= 0:
            continue
        dotlocs = random.sample(dotlocopts, ndots)
        dotcols = random.sample(allcols, ndots)
        for dotlocj, col in zip(dotlocs, dotcols):
            gi = fill(gi, col, {(ii, dotlocj)})
            if col in linecols:
                idx = linecols.index(col)
                linelocj = locs[idx]
                if dotlocj > linelocj:
                    go = fill(go, col, {(ii, linelocj + 1)})
                else:
                    go = fill(go, col, {(ii, linelocj - 1)})

    if mirror:
        gi = dmirror(gi)
        go = dmirror(go)

    return {'input': gi, 'output': output_guard(go)} if False else {'input': gi, 'output': go}


# --------------------------------------------------------------------------
# 3. derive_operations
# --------------------------------------------------------------------------
# Rule (gravitate): the grid holds full-grid lines (uniform columns, or uniform
# rows after the dmirror) plus scattered single-cell dots.
#   * a dot whose colour equals a line's colour SLIDES along its row/column
#     until it rests directly beside that line, on the side it came from.
#   * every other dot is noise and is wiped to background.
# So: erase the noise, then MOVE each matching dot to its line.
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # background: the canvas colour the generator paints before drawing anything.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- find the full-grid lines -----------------------------------------
    row_lines, col_lines = {}, {}
    for r in range(h):
        v = int(I[r, 0])
        if v != bgc and bool(np.all(I[r, :] == v)):
            row_lines[r] = v
    for c in range(w):
        v = int(I[0, c])
        if v != bgc and bool(np.all(I[:, c] == v)):
            col_lines[c] = v

    vertical = len(col_lines) >= len(row_lines) and len(col_lines) > 0
    lines = col_lines if vertical else row_lines
    if not lines:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels
    pos_of = {col: p for p, col in lines.items()}

    # --- collect the dots and work out where each one gravitates to -------
    movers = []       # (src, dst, colour)  -> performed with Move ops
    zero_dots = []    # (src, dst, 0)       -> colour 0 cannot be grabbed by Move
    noise = []        # cells to wipe

    for r in range(h):
        for c in range(w):
            v = int(I[r, c])
            if v == bgc:
                continue
            if vertical and c in lines:
                continue
            if (not vertical) and r in lines:
                continue
            if v in pos_of:
                p = pos_of[v]
                if vertical:
                    dst = (r, p + 1 if c > p else p - 1)
                else:
                    dst = (p + 1 if r > p else p - 1, c)
                if (r, c) == dst:
                    continue                      # already resting on the line
                if v == 0:
                    zero_dots.append(((r, c), dst, v))
                else:
                    movers.append(((r, c), dst, v))
            else:
                noise.append((r, c))

    # --- step 1: wipe every dot that belongs to no line -------------------
    # (colour-0 dots are wiped here too: ARCLE's object mode treats 0 as
    #  "nothing", so they cannot be grabbed and are re-drawn at the end.)
    erase = sorted(set(noise) | {s for (s, d, v) in zero_dots})
    if erase:
        ops.append(int(bgc)); sels.append(sel_of(erase))

    # --- step 2: order the slides so no dot is crushed before it moves ----
    remaining = set(range(len(movers)))
    order = []
    forced_paint = []
    while remaining:
        blocked_srcs = {movers[j][0] for j in remaining}
        ready = [i for i in remaining
                 if movers[i][1] not in (blocked_srcs - {movers[i][0]})]
        if ready:
            order.extend(ready)
            remaining -= set(ready)
        else:
            # cyclic swap: move one, and re-draw the dot it lands on
            i = sorted(remaining)[0]
            order.append(i)
            remaining.discard(i)
            tgt = movers[i][1]
            for j in sorted(remaining):
                if movers[j][0] == tgt:
                    remaining.discard(j)
                    forced_paint.append(movers[j])

    # --- step 3: slide each dot to its line -------------------------------
    for i in order:
        (sr, sc), (dr_, dc_), v = movers[i]
        drow, dcol = dr_ - sr, dc_ - sc
        if drow != 0:
            op = 20 if drow < 0 else 21
            steps = abs(drow)
        else:
            op = 23 if dcol < 0 else 22
            steps = abs(dcol)
        ops.append(op); sels.append(sel_of([(sr, sc)]))      # grab the dot
        for _ in range(steps - 1):
            ops.append(op); sels.append(sel_of([]))          # keep it gliding

    # --- step 4: repair the footprints the dots left behind ---------------
    moved_srcs = {movers[i][0] for i in order}
    moved_dsts = {movers[i][1] for i in order}
    hole = sorted(moved_srcs - moved_dsts)
    if bgc != 0 and hole:
        ops.append(int(bgc)); sels.append(sel_of(hole))

    # --- step 5: draw the dots that could not be grabbed ------------------
    bycol = {}
    for (s, d, v) in zero_dots:
        bycol.setdefault(int(v), []).append(d)
    for (s, d, v) in forced_paint:
        bycol.setdefault(int(v), []).append(d)
    for v in sorted(bycol):
        cells = sorted(set(bycol[v]))
        if cells:
            ops.append(int(v)); sels.append(sel_of(cells))

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
    return ops, sels


# ── GridMaker ─────────────────────────────────────────────────────────────────

class GridMaker(BaseGridMaker):

    def parse(self, **kwargs) -> List[Tuple[
        List[NDArray], List[NDArray],
        List[NDArray], List[NDArray],
        Dict[str, Any],
    ]]:
        num_samples  = kwargs.get("num_samples", 1)
        num_examples = kwargs.get("num_examples", 3)
        max_h, max_w = kwargs.get("max_grid_dim", [30, 30])
        dataset = []

        for _sn in range(num_samples):
            # Episode-level retry: if 10 attempts at some instance all fail, that's
            # transient (bad luck with the generator's randomness) — retry the WHOLE
            # episode from scratch (fresh colors/instance plan) up to 5 times, rather
            # than silently continuing with a partial episode (fewer examples than
            # requested, or a missing test instance with operations=[]/selections=[]
            # quietly appended as if it were a normal sample).
            for _episode_attempt in range(5):
                pr_in:  List[NDArray] = []
                pr_out: List[NDArray] = []
                ex_in:  List[NDArray] = []
                ex_out: List[NDArray] = []
                ops:  List[int]       = []
                sels: List[List[int]] = []

                # sample color roles once per episode → consistent across all instances
                # sample_colors() may optionally accept num_examples (to pre-plan
                # per-instance categories) — call it either way for compatibility
                # with grid_makers generated before this parameter existed.
                if "num_examples" in inspect.signature(sample_colors).parameters:
                    colors = sample_colors(num_examples=num_examples)
                else:
                    colors = sample_colors()

                # Plans are consumed by INDEX, not mutated: retries for instance j
                # must receive the same variant. category_plan is retained as a
                # backwards-compatible single-key form; new makers use kwargs dict entries.
                category_plan = colors.pop("category_plan", None) if isinstance(colors, dict) else None
                instance_plan = colors.pop("instance_plan", None) if isinstance(colors, dict) else None
                if category_plan is not None and instance_plan is not None:
                    raise ValueError(
                        "sample_colors must return only one of category_plan/instance_plan"
                    )
                if category_plan is not None and len(category_plan) != num_examples + 1:
                    # A wrong plan length is a deterministic bug in sample_colors(),
                    # not bad luck — retrying the episode won't fix it. Fail loudly
                    # instead of clamping the index and silently reusing an entry.
                    raise ValueError(
                        f"category_plan length {len(category_plan)} != "
                        f"num_examples+1 ({num_examples + 1}) for task 1a07d186"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1a07d186"
                        )
                    if any(not isinstance(entry, dict) for entry in instance_plan):
                        raise ValueError("every instance_plan entry must be a kwargs dict")
                    if instance_plan[-1] not in instance_plan[:-1]:
                        raise ValueError(
                            "instance_plan test variant must appear among the examples"
                        )

                try:
                    j = 0
                    while j < num_examples + 1:
                        ok = False
                        for _ in range(10):
                            try:
                                call_kwargs = dict(colors)
                                if instance_plan is not None:
                                    call_kwargs.update(instance_plan[j])
                                elif category_plan is not None:
                                    call_kwargs["category"] = category_plan[j]
                                r = generate(
                                    random.uniform(0.2, 0.5),
                                    random.uniform(0.5, 0.8),
                                    max_h, max_w,
                                    **call_kwargs,
                                )
                                I = np.array(r["input"],  dtype=np.uint8)
                                O = np.array(r["output"], dtype=np.uint8)
                                # enforce max_grid_dim — skip oversized grids
                                if I.shape[0] > max_h or I.shape[1] > max_w:
                                    continue
                                if O.shape[0] > max_h or O.shape[1] > max_w:
                                    continue
                                ok = True
                                break
                            except (IndexError, ValueError, KeyError):
                                continue
                        if not ok:
                            raise RuntimeError(
                                f"Failed to generate instance {j} after 10 attempts "
                                f"for task 1a07d186"
                            )
                        if j == num_examples:
                            pr_in.append(I)
                            pr_out.append(O)
                            ops, sels = derive_operations(I, O)
                        else:
                            ex_in.append(I)
                            ex_out.append(O)
                        j += 1
                    break  # episode complete
                except RuntimeError:
                    continue
            else:
                raise RuntimeError(
                    f"Failed to build a complete episode for task 1a07d186 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1a07d186-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
