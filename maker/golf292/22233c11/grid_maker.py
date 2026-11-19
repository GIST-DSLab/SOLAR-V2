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


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ccols=None) -> dict:
    cols = [c for c in range(10) if c != 8]
    if bgc is None:
        bgc = random.choice(cols)
    if not ccols:
        remcols = [c for c in cols if c != bgc]
        ccols = random.sample(remcols, random.randint(1, len(remcols)))

    def unif(lo, hi):
        hi = max(lo, hi)
        d = random.uniform(diff_lb, diff_ub)
        return int(round(lo + d * (hi - lo)))

    h = unif(5, max_h)
    w = unif(5, max_w)
    nobjs = unif(1, max(1, (h * w) // 10))
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    inds = {(r, c) for r in range(h) for c in range(w)}
    fullinds = set(inds)
    succ, tr, maxtr = 0, 0, 10 * nobjs
    while succ < nobjs and tr < maxtr:
        if not inds:
            break
        tr += 1
        od = random.randint(1, 3)
        col = random.choice(ccols)
        flip = random.choice((True, False))
        g = [[bgc] * 4 for _ in range(4)]
        g[0][3] = 8; g[3][0] = 8
        g[1][1] = col; g[2][2] = col
        if flip:
            g = g[::-1]
        inobj, outobj = [], []
        for i in range(4):
            for j in range(4):
                v = g[i][j]
                if v == bgc:
                    continue
                for a in range(od):
                    for b in range(od):
                        cell = (i * od + a, j * od + b, v)
                        outobj.append(cell)
                        if v == col:
                            inobj.append(cell)
        lr, lc = random.choice(list(inds))
        inobj = [(r + lr, c + lc, v) for r, c, v in inobj]
        outobj = [(r + lr, c + lc, v) for r, c, v in outobj]
        in_idx = {(r, c) for r, c, _ in inobj}
        out_idx = {(r, c) for r, c, _ in outobj}
        if in_idx.issubset(inds) and (out_idx & fullinds).issubset(inds):
            succ += 1
            nb = set()
            for r, c in out_idx:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nb.add((r + dr, c + dc))
            inds = (inds - out_idx) - nb
            for r, c, v in inobj:
                gi[r, c] = v
            for r, c, v in outobj:
                if 0 <= r < h and 0 <= c < w:
                    go[r, c] = v
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    """Each object = two od x od squares touching at a corner, sitting as blocks
    (1,1),(2,2) or (1,2),(2,1) of a 4x4-block frame.  The rule marks the two frame
    corners on the OTHER diagonal with 8.  All marker blocks are determined by the
    input objects alone, so they are painted in one Color8 stroke."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # row 0 / col 0 always lie outside every object (objects start one block
    # inside their frame), so they are pure canvas
    bgc = int(I[0, 0])

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

    def block(br, bc, od):
        return [(rr, cc)
                for rr in range(br, br + od)
                for cc in range(bc, bc + od)
                if 0 <= rr < hi and 0 <= cc < wi]

    targets = set()
    for cells in comps:
        r0 = min(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        r1 = max(r for r, _ in cells)
        od = max(1, (r1 - r0 + 1) // 2)
        R0, C0 = r0 - od, c0 - od
        if I[r0, c0] != bgc:          # "\" diagonal -> mark "/" corners
            tgt = block(R0, C0 + 3 * od, od) + block(R0 + 3 * od, C0, od)
        else:                         # "/" diagonal -> mark "\" corners
            tgt = block(R0, C0, od) + block(R0 + 3 * od, C0 + 3 * od, od)
        targets.update(t for t in tgt if I[t] != 8)

    if targets:
        ops.append(8)
        sels.append(sel_of(sorted(targets)))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
