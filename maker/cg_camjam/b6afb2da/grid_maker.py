"""
ARC Task: b6afb2da (RE-ARC) — LLM-generated grid_maker
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
from collections import Counter

import numpy as np
from maker.sel_helpers import sel_of


# ───────────────────────── colour roles ─────────────────────────
def sample_colors(num_examples=None) -> dict:
    # 1, 2 and 4 are reserved for the output paint and are never background
    cols = [c for c in range(10) if c not in (1, 2, 4)]
    bgc = random.choice(cols)
    return {"bgc": bgc}


# ───────────────────────── generator ─────────────────────────
def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return int(round(a + (b - a) * d))

    cols = [c for c in range(10) if c not in (1, 2, 4)]
    if bgc is None:
        bgc = random.choice(cols)
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    h, w = min(h, max_h), min(w, max_w)
    remcols = [c for c in cols if c != bgc]
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    num = unifint(diff_lb, diff_ub, (1, 9))
    indss = {(i, j) for i in range(h) for j in range(w)}
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or len(indss) == 0:
            break
        oh = random.randint(3, 7)
        ow = random.randint(3, 7)
        subs = [ij for ij in indss if ij[0] < h - oh and ij[1] < w - ow]
        if not subs:
            tr += 1
            continue
        loci, locj = random.choice(subs)
        bd = {(i, j) for i in range(loci, loci + oh) for j in range(locj, locj + ow)}
        col = random.choice(remcols)
        if bd.issubset(indss):
            remcols = [c for c in remcols if c != col]
            for (i, j) in bd:
                gi[i, j] = col
                go[i, j] = 2
            for (i, j) in bd:
                if i in (loci, loci + oh - 1) or j in (locj, locj + ow - 1):
                    go[i, j] = 4
            for (i, j) in [(loci, locj), (loci, locj + ow - 1),
                           (loci + oh - 1, locj), (loci + oh - 1, locj + ow - 1)]:
                go[i, j] = 1
            succ += 1
            indss -= bd
        tr += 1
    return {"input": gi.tolist(), "output": go.tolist()}


# ───────────────────────── helpers ─────────────────────────
def _components(G):
    hh, ww = G.shape
    seen = np.zeros((hh, ww), dtype=bool)
    out = []
    for r in range(hh):
        for c in range(ww):
            if seen[r, c]:
                continue
            col = G[r, c]
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hh and 0 <= nx < ww and not seen[ny, nx] and G[ny, nx] == col:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            out.append((int(col), cells))
    return out


def _bg_of(G):
    return int(Counter(G.flatten().tolist()).most_common(1)[0][0])


# ───────────────────────── trajectory ─────────────────────────
def derive_operations(I, O, examples=None):
    """
    Reading (the golf entry's): every planted rectangle gets one base fill of 2;
    then the four sides are visited one after another (top, right, bottom,
    left) across ALL rectangles together. A side cell touched for the first
    time turns 2 -> 4; a cell touched by a second side (a corner, where two
    sides meet) turns 4 -> 1. Each side pass therefore splits into two masks:
    the side cells that still show 2 and those that already show 4 — both are
    read from the grid as it currently stands.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: the colour the canvas is painted with (same in every demo)
    bgc = _bg_of(I)
    if examples:
        votes = []
        for pair in examples:
            try:
                votes.append(_bg_of(np.asarray(pair[0], dtype=int)))
            except Exception:
                continue
        if votes:
            cons = Counter(votes).most_common(1)[0][0]
            if cons in set(I.flatten().tolist()):
                bgc = int(cons)

    # rectangles = solid single-colour non-background components
    rects = []
    for col, cells in _components(I):
        if col == bgc:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, c0 = min(rs), min(cs)
        h = max(rs) - r0 + 1
        w = max(cs) - c0 + 1
        if len(cells) == h * w:
            rects.append((r0, c0, h, w))
    rects.sort()

    ops, sels = [], []
    if not rects:
        ops.append(34); sels.append([0, 0, hi - 1, wi - 1])
        return ops, sels

    G = I.copy()

    # 1) base layer: every rectangle's body becomes 2
    body = [(r, c) for (r0, c0, h, w) in rects
            for r in range(r0, r0 + h) for c in range(c0, c0 + w)]
    ops.append(2); sels.append(sel_of(body))
    for (r, c) in body:
        G[r, c] = 2

    # 2) the four sides in turn, across all rectangles at once
    def side_cells(rect, side):
        r0, c0, h, w = rect
        if side == "top":
            return [(r0, c) for c in range(c0, c0 + w)]
        if side == "right":
            return [(r, c0 + w - 1) for r in range(r0, r0 + h)]
        if side == "bottom":
            return [(r0 + h - 1, c) for c in range(c0, c0 + w)]
        return [(r, c0) for r in range(r0, r0 + h)]

    for side in ("top", "right", "bottom", "left"):
        cells = [p for rect in rects for p in side_cells(rect, side)]
        second = [p for p in cells if G[p] == 4]   # already on a side -> corner
        first = [p for p in cells if G[p] == 2]    # first side touching it
        if second:
            ops.append(1); sels.append(sel_of(second))
            for p in second:
                G[p] = 1
        if first:
            ops.append(4); sels.append(sel_of(first))
            for p in first:
                G[p] = 4

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
                        f"num_examples+1 ({num_examples + 1}) for task b6afb2da"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b6afb2da"
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
                                f"for task b6afb2da"
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
                    f"Failed to build a complete episode for task b6afb2da "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b6afb2da-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
