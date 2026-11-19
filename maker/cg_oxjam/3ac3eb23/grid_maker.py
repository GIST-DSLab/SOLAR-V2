"""
ARC Task: 3ac3eb23 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # Only the background is an episode-level role; seed colours are free
    # (the rule only depends on where the seeds are, not which colour they are).
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        lo = a + round((b - a) * lb)
        hi_ = a + round((b - a) * ub)
        lo, hi_ = min(lo, hi_), max(lo, hi_)
        return random.randint(max(a, lo), min(b, max(a, hi_)))

    if bgc is None:
        bgc = random.choice(range(10))
    cols = list(range(10))
    remcols = [c for c in cols if c != bgc]

    # Pick the rotation first so the rotated grid still fits within max_h x max_w.
    mf = random.choice([0, 1, 2, 3])
    if mf in (1, 3):
        hb, wb = max_w, max_h
    else:
        hb, wb = max_h, max_w
    h = unifint(diff_lb, diff_ub, (3, min(30, hb)))
    w = unifint(diff_lb, diff_ub, (3, min(30, wb)))

    nlocs = unifint(diff_lb, diff_ub, (1, max(1, (w - 2) // 3)))
    locopts = list(range(1, w - 1))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    for _ in range(nlocs):
        if len(locopts) == 0:
            break
        locj = random.choice(locopts)
        locopts = [x for x in locopts if not (locj - 2 <= x < locj + 3)]
        col = random.choice(remcols)
        gi[0, locj] = col
        for p in range(0, h, 2):
            go[p, locj] = col
        for p in range(1, h, 2):
            go[p, locj - 1] = col
            go[p, locj + 1] = col
    # rot90 (CW) = np.rot90(k=3); rot180 = k=2; rot270 = k=1
    kmap = {0: 0, 1: 3, 2: 2, 3: 1}
    gi = np.rot90(gi, kmap[mf])
    go = np.rot90(go, kmap[mf])
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """
    How I read the task (the same reading as the golf entry): the border line that
    holds the seeds is one unit. Its partner line, one step in, is that line spread
    sideways by one cell: every seed gets two wings. The output is this 2-line band
    repeated with period 2 until it reaches the far border.
    Steps:
      (1) grow the wings: one Color op per seed colour, covering all its wing cells
      (2) copy the WHOLE 2-line band, border to border, and paste it every 2 lines
      (3) an odd strip leaves one last line: copy the seed line and paste it there
      (4) colour-0 seeds are invisible to Copy/Paste, so paint their fans directly
    Nothing here reads O.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # bgc: the colour that fills the canvas; seeds are a few isolated border cells
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    seeds = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]

    ops, sels = [], []
    if not seeds:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    # which border holds the seeds: u = inward step, D = depth of the strip
    if all(r == 0 for r, _ in seeds) and hi > 1:
        u, v, D, side = (1, 0), (0, 1), hi, "top"
    elif all(r == hi - 1 for r, _ in seeds):
        u, v, D, side = (-1, 0), (0, 1), hi, "bottom"
    elif all(c == 0 for _, c in seeds):
        u, v, D, side = (0, 1), (1, 0), wi, "left"
    else:
        u, v, D, side = (0, -1), (1, 0), wi, "right"

    def cell(s, d, t):
        return (s[0] + u[0] * d + v[0] * t, s[1] + u[1] * d + v[1] * t)

    def line_sel(d0, n):
        # full-width (or full-height) block of n lines starting at depth d0,
        # given by its top-left corner and size: a whole rectangle on purpose
        if side == "top":
            return [d0, 0, n - 1, wi - 1]
        if side == "bottom":
            return [hi - d0 - n, 0, n - 1, wi - 1]
        if side == "left":
            return [0, d0, hi - 1, n - 1]
        return [0, wi - d0 - n, hi - 1, n - 1]

    def origin(d0, n):
        s = line_sel(d0, n)
        return [s[0], s[1], 0, 0]

    # (1) wings at depth 1, one Color op per seed colour
    by_col = {}
    for s in seeds:
        col = int(I[s[0], s[1]])
        by_col.setdefault(col, []).extend([cell(s, 1, -1), cell(s, 1, 1)])
    for col in sorted(by_col):
        ops.append(col); sels.append(sel_of(by_col[col]))

    # (2) + (3) repeat the 2-line band (whole width) every 2 lines
    visible = any(int(I[s[0], s[1]]) != 0 for s in seeds)  # Copy/Paste needs nonzero content
    P = D // 2
    pasted = False
    if visible and P > 1:
        ops.append(29); sels.append(line_sel(0, 2))         # CopyO the band: whole rectangle
        for k in range(1, P):
            ops.append(30); sels.append(origin(2 * k, 2))
        pasted = True
    if visible and D % 2 == 1 and D >= 3:
        ops.append(29); sels.append(line_sel(0, 1))         # CopyO the seed line: whole line
        ops.append(30); sels.append(origin(D - 1, 1))
        pasted = True

    # (4) seeds Copy/Paste could not carry: draw the rest of their fan directly
    for s in seeds:
        col = int(I[s[0], s[1]])
        if col != 0 and pasted:
            continue
        if col != 0 and not visible:
            continue
        cells = []
        for d in range(2, D):
            for t in ((0,) if d % 2 == 0 else (-1, 1)):
                cells.append(cell(s, d, t))
        if cells:
            ops.append(col); sels.append(sel_of(cells))

    ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 3ac3eb23"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3ac3eb23"
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
                                f"for task 3ac3eb23"
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
                    f"Failed to build a complete episode for task 3ac3eb23 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3ac3eb23-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
