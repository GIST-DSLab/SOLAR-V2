"""
ARC Task: 444801d8 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    numcols = random.randint(2, 9)
    ccols = random.sample(remcols, numcols)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi)))))

    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    if ccols is None:
        remcols = [c for c in cols if c != bgc]
        ccols = random.sample(remcols, random.randint(2, 9))
    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 25)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    succ, tr, maxtr = 0, 0, 5 * nobjs
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = random.randint(4, 6)
        ow = 5
        # box outline rows 1..oh-1, cols 0..4, gap at (1,2)
        bx = set()
        for r in range(1, oh):
            for c in range(0, 5):
                if r in (1, oh - 1) or c in (0, 4):
                    bx.add((r, c))
        bx.discard((1, 2))
        fullobj = {(r, c) for r in range(0, oh) for c in range(0, 5)}
        cands = [(r, c) for r in range(1, oh) for c in range(0, 5) if (r, c) not in bx]
        dot = random.choice(cands)
        dcol, bxcol = random.sample(ccols, 2)
        inobj = {(bxcol, p) for p in bx} | {(dcol, dot)}
        outobj = {(bxcol, p) for p in bx} | {(dcol, p) for p in fullobj - bx}
        if random.choice((True, False)):
            # hmirror over rows 0..oh-1 then shift up for input
            inobj = {(v, (oh - 1 - r - 1, c)) for v, (r, c) in inobj}
            outobj = {(v, (oh - 1 - r, c)) for v, (r, c) in outobj}
        cl = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cl:
            continue
        lr, lc = random.choice(cl)
        outplcd = {(v, (r + lr, c + lc)) for v, (r, c) in outobj}
        outi = {p for _, p in outplcd}
        if outi.issubset(inds):
            succ += 1
            inplcd = {(v, (r + lr, c + lc)) for v, (r, c) in inobj}
            ini = {p for _, p in inplcd}
            rs = [r for r, _ in ini]; cs = [c for _, c in ini]
            r0, r1, c0, c1 = min(rs) - 1, max(rs) + 1, min(cs) - 1, max(cs) + 1
            ob = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                  if r in (r0, r1) or c in (c0, c1)}
            inds = (inds - outi) - ob
            for v, (r, c) in inplcd:
                if 0 <= r < h and 0 <= c < w:
                    gi[r][c] = v
            for v, (r, c) in outplcd:
                if 0 <= r < h and 0 <= c < w:
                    go[r][c] = v
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """
    Same rule (each gapped box: hollow + line beyond the gap take the dot's colour),
    but the unit of work is a COLOUR: for each dot colour, one Color op paints every
    hollow and every outgoing line of all boxes whose dot has that colour, at once.
    All selections are measured from I only.
    """
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = int(I[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append((col, cells))

    boxes = [(col, cells) for col, cells in comps if len(cells) > 1]
    boxes.sort(key=lambda t: (min(r for r, _ in t[1]), min(c for _, c in t[1])))

    groups = {}   # dot colour -> list of cells to paint
    order = []
    for col, cells in boxes:
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        cellset = set(cells)
        hollow = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                  if (r, c) not in cellset]
        if not hollow:
            continue
        cnt = Counter(int(I[r, c]) for r, c in hollow)
        dcol = min(cnt.items(), key=lambda kv: (kv[1], kv[0]))[0]
        gap_on_top = any(r == r0 for r, _ in hollow)
        line_r = r0 - 1 if gap_on_top else r1 + 1
        region = [p for p in hollow if I[p] != dcol]   # dot already holds dcol
        if 0 <= line_r < h:
            region += [(line_r, c) for c in range(c0, c1 + 1) if I[line_r, c] != dcol]
        if dcol not in groups:
            groups[dcol] = []
            order.append(dcol)
        groups[dcol].extend(region)

    ops, sels = [], []
    for dcol in order:
        cells = sorted(set(groups[dcol]))
        if cells:
            ops.append(dcol)
            sels.append(sel_of(cells))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 444801d8"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 444801d8"
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
                                f"for task 444801d8"
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
                    f"Failed to build a complete episode for task 444801d8 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"444801d8-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
