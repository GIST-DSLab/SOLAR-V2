"""
ARC Task: b91ae062 (RE-ARC) — LLM-generated grid_maker
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
    """Episode-level palette + a per-instance plan for the number of colours.

    The rule is 'upscale by (number of colours - 1)', so the *count* of colours is the
    structural variant here: the examples must show several different counts (and the
    test must reuse one of them) or the factor is not learnable.  The palette itself is
    fixed once per episode (nested prefixes), so every instance is drawn in the same
    colour scheme.  0 is deliberately made an ordinary picture colour about half the
    time, so routes that carry cells around must cope with it.
    """
    cols = list(range(10))
    random.shuffle(cols)
    if random.random() < 0.5:
        i = cols.index(0)
        j = random.randrange(3)
        cols[i], cols[j] = cols[j], cols[i]

    values = [3, 4, 5, 6, 7]
    n_ex = num_examples if num_examples else 3
    if n_ex <= len(values):
        ex_vals = random.sample(values, n_ex)
    else:
        ex_vals = list(values) + [random.choice(values) for _ in range(n_ex - len(values))]
    random.shuffle(ex_vals)

    plan = [{"numc": v} for v in ex_vals]
    plan.append({"numc": random.choice(ex_vals)})   # test case was demonstrated
    return {"palette": cols, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, palette, numc=None) -> dict:
    cols = list(palette)
    if numc is None:
        numc = random.choice([3, 4, 5, 6, 7])
    numc = max(3, min(int(numc), 10, len(cols)))

    k = numc - 1
    hub = max(2, min(5, max_h // k))
    wub = max(2, min(5, max_w // k))

    h = w = None
    for _ in range(64):
        hh = unifint(diff_lb, diff_ub, (2, hub))
        ww = unifint(diff_lb, diff_ub, (2, wub))
        if hh * ww >= numc:
            h, w = hh, ww
            break
    if h is None:
        h, w = hub, wub

    # make sure the upscaled output still fits the allowed canvas
    numc = max(3, min(numc, h * w, 1 + min(max_h // h, max_w // w)))

    ccols = list(cols[:numc])                  # consistent colour scheme across the episode
    c = canvas(-1, (h, w))
    inds = totuple(asindices(c))
    fixinds = random.sample(list(inds), numc)  # every colour appears at least once
    fixset = set(fixinds)
    shuffled = list(ccols)
    random.shuffle(shuffled)
    obj = {(cc, ij) for cc, ij in zip(shuffled, fixinds)}
    for ij in inds:
        if ij not in fixset:
            obj.add((random.choice(ccols), ij))

    gi = paint(c, obj)
    go = upscale(gi, numc - 1)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """Rule (read off I alone): every cell becomes a k x k block of its own colour,
    with k = (number of distinct colours in I) - 1.

    O is never inspected.  Which *route* is taken through that rule -- where the sweep
    starts, whether colours are handled cell by cell or all at once, whether a base
    colour is laid down first, whether a repeated block is stamped from the first one --
    is decided from features of I (its shape, its dominant colour and where that colour
    first appears, its corner colours), so one instance always produces one route and
    the variety comes from the instances differing.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    k = len(np.unique(I)) - 1
    ho, wo = hi * k, wi * k

    ops, sels = [], []

    # Grow the working canvas to the upscaled size.  ResizeGrid transparent-copies I into
    # the top-left corner; everything else comes up 0.  (Full rectangle -> bbox form.)
    ops.append(33)
    sels.append([0, 0, ho - 1, wo - 1])

    cur = np.zeros((ho, wo), dtype=int)
    cur[:hi, :wi] = I

    def block_cells(i, j):
        r0, c0 = i * k, j * k
        return [(r0 + di, c0 + dj) for di in range(k) for dj in range(k)]

    def paint_cells(cells, col):
        """Color(col) over exactly these cells; emitted only if it visibly changes the grid.
        col may be 0 -- then this is the explicit Color0 that puts those cells down."""
        if not cells:
            return False
        if all(cur[r, c] == col for r, c in cells):
            return False
        ops.append(int(col))
        sels.append(sel_of(cells))
        for r, c in cells:
            cur[r, c] = col
        return True

    # ---- features of I that decide the route -------------------------------------
    flat = [(int(I[r, c]), r, c) for r in range(hi) for c in range(wi)]
    cnt = Counter(v for v, _, _ in flat)
    first = {}
    for v, r, c in flat:
        first.setdefault(v, (r, c))
    mx_cnt = max(cnt.values())
    dom = min((v for v in cnt if cnt[v] == mx_cnt), key=lambda v: first[v])
    dr, dc = first[dom]

    # sweep along the longer side; for a square picture along the side the dominant
    # colour leans to.  Start from the corner nearest that colour's first appearance.
    if wi > hi:
        axis = 'row'
    elif hi > wi:
        axis = 'col'
    else:
        axis = 'row' if dr <= dc else 'col'
    rows = list(range(hi))[::-1] if dr * 2 >= hi else list(range(hi))
    colz = list(range(wi))[::-1] if dc * 2 >= wi else list(range(wi))

    order = []
    if axis == 'row':
        for r in rows:
            for c in colz:
                order.append((r, c))
    else:
        for c in colz:
            for r in rows:
                order.append((r, c))

    col_order = []
    cells_of = {}
    pos_of = {}
    for (r, c) in order:
        v = int(I[r, c])
        if v not in cells_of:
            col_order.append(v)
            cells_of[v] = []
            pos_of[v] = []
        cells_of[v].extend(block_cells(r, c))
        pos_of[v].append((r, c))

    corner = int(I[0, 0]) + int(I[0, wi - 1]) + int(I[hi - 1, 0]) + int(I[hi - 1, wi - 1])
    fams = ['blocks', 'groups']
    if mx_cnt >= 2:
        fams.append('base')
    if any(v != 0 and cnt[v] >= 2 for v in cnt):
        fams.append('stamp')
    fam = fams[corner % len(fams)]

    # ---- the route ----------------------------------------------------------------
    if fam == 'blocks':
        # one block at a time, walking the picture in the chosen sweep direction
        for (r, c) in order:
            paint_cells(block_cells(r, c), int(I[r, c]))

    elif fam == 'groups':
        # deal with each colour as one thing: all of its blocks in a single stroke,
        # colours taken in the order the sweep first meets them
        for v in col_order:
            paint_cells(cells_of[v], v)

    elif fam == 'base':
        # lay the dominant colour down over the whole canvas, then draw the rest on top
        if not np.all(cur == dom):
            ops.append(int(dom))
            sels.append([0, 0, ho - 1, wo - 1])   # full rectangle: the whole canvas
            cur[:, :] = dom
        for v in col_order:
            if v == dom:
                continue
            paint_cells(cells_of[v], v)

    else:  # 'stamp'
        # a colour that occurs several times: make its first block, then copy that block
        # onto its other places.  0 cannot be copied (ARCLE reads it as nothing there),
        # so 0-blocks -- and one-off colours -- are put down with a plain Color instead.
        for v in col_order:
            places = pos_of[v]
            if v == 0 or len(places) < 2:
                paint_cells(cells_of[v], v)
                continue
            sr, sc = places[0]
            paint_cells(block_cells(sr, sc), v)
            todo = [(r, c) for (r, c) in places[1:]
                    if not all(cur[rr, cc] == v for rr, cc in block_cells(r, c))]
            if not todo:
                continue
            ops.append(29)                                    # CopyO the finished block
            sels.append([sr * k, sc * k, k - 1, k - 1])       # full rectangle: that block
            for (r, c) in todo:
                ops.append(30)
                sels.append([r * k, c * k, 0, 0])             # paste origin
                for rr, cc in block_cells(r, c):
                    cur[rr, cc] = v

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task b91ae062"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b91ae062"
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
                                f"for task b91ae062"
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
                    f"Failed to build a complete episode for task b91ae062 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b91ae062-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
