"""
ARC Task: 22233c11 (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    ncols = random.randint(1, len(remcols))
    ccols = random.sample(remcols, ncols)
    return {"bgc": bgc, "ccols": ccols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    cols = [c for c in range(10) if c != 8]
    if bgc is None:
        bgc = random.choice(cols)
    if not ccols:
        rem = [c for c in cols if c != bgc]
        ccols = random.sample(rem, random.randint(1, len(rem)))
    ccols = [c for c in ccols if c != bgc and c != 8] or [c for c in cols if c != bgc][:1]

    h = unifint(diff_lb, diff_ub, (5, max(5, max_h)))
    w = unifint(diff_lb, diff_ub, (5, max(5, max_w)))
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 10)))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    fullinds = {(r, c) for r in range(h) for c in range(w)}
    inds = set(fullinds)
    succ, tr, maxtr = 0, 0, 10 * nobjs
    while succ < nobjs and tr < maxtr:
        if not inds:
            break
        tr += 1
        od = random.randint(1, 3)
        g = [[None] * 4 for _ in range(4)]
        g[0][3] = 8
        g[3][0] = 8
        col = random.choice(ccols)
        g[1][1] = col
        g[2][2] = col
        if random.choice((True, False)):
            g = g[::-1]
        inobj, outobj = set(), set()
        for i in range(4):
            for j in range(4):
                v = g[i][j]
                if v is None:
                    continue
                for a in range(od):
                    for b in range(od):
                        cell = (v, (i * od + a, j * od + b))
                        outobj.add(cell)
                        if v == col:
                            inobj.add(cell)
        loc = random.choice(sorted(inds))
        outobj = {(v, (r + loc[0], c + loc[1])) for v, (r, c) in outobj}
        inobj = {(v, (r + loc[0], c + loc[1])) for v, (r, c) in inobj}
        outi = {p for _, p in outobj}
        ini = {p for _, p in inobj}
        if ini.issubset(inds) and (outi & fullinds).issubset(inds):
            succ += 1
            nb = set()
            for (r, c) in outi:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nb.add((r + dr, c + dc))
            inds = inds - outi - nb
            for v, (r, c) in inobj:
                gi[r][c] = v
            for v, (r, c) in outobj:
                if 0 <= r < h and 0 <= c < w:
                    go[r][c] = v
    return {"input": gi, "output": go}


def derive_operations(I, O):
    """Reading (verifier / golf): each object is a diagonal of two od-squares.
    Mirror it inside its own 2od x 2od box (the two EMPTY blocks of that box,
    painted 8 = the mirrored object), then scale that mirrored copy x2 about
    the object's centre: each 8 block slides od cells diagonally outward
    (vertical then horizontal unit Moves, one grab per block)."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    bgc = int(I[0, 0])  # row 0 is always pure canvas (objects start one block inside frame)

    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < hi and 0 <= nc < wi and not seen[nr, nc] \
                                    and I[nr, nc] != bgc:
                                seen[nr, nc] = True
                                stack.append((nr, nc))
                comps.append(cells)
    comps.sort(key=lambda cs: (min(r for r, _ in cs), min(c for _, c in cs)))

    for cells in comps:
        r0 = min(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        r1 = max(r for r, _ in cells)
        od = max(1, (r1 - r0 + 1) // 2)
        cellset = set(cells)

        # the two empty blocks of the object's 2x2 block box = mirrored object
        blocks = []
        for bi in range(2):
            for bj in range(2):
                br, bc = r0 + bi * od, c0 + bj * od
                if (br, bc) in cellset:
                    continue
                blk = [(br + a, bc + b) for a in range(od) for b in range(od)]
                dr = -1 if bi == 0 else 1
                dc = -1 if bj == 0 else 1
                dest = [(r + dr * od, c + dc * od) for r, c in blk]
                if not any(0 <= r < hi and 0 <= c < wi for r, c in dest):
                    continue  # scaled block lands wholly off-grid: nothing to draw
                blocks.append((blk, dr, dc))
        if not blocks:
            continue

        # mirror: paint the object's opposite diagonal with 8
        mirror_cells = sorted(set(p for blk, _, _ in blocks for p in blk))
        ops.append(8)
        sels.append(sel_of(mirror_cells))

        # scale x2 about centre: slide each mirrored block od cells outward
        for blk, dr, dc in blocks:
            steps = [20 if dr < 0 else 21] * od + [23 if dc < 0 else 22] * od
            for k, op in enumerate(steps):
                ops.append(op)
                sels.append(sel_of(blk) if k == 0 else sel_of([]))
            if bgc != 0:
                # destination is disjoint from the start block: whole start footprint is vacated
                ops.append(bgc)
                sels.append(sel_of(blk))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])  # full-grid bbox for Submit
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
                        f"num_examples+1 ({num_examples + 1}) for task 22233c11"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 22233c11"
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
                                f"for task 22233c11"
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
                    f"Failed to build a complete episode for task 22233c11 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"22233c11-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
