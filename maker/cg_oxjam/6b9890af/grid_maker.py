"""
ARC Task: 6b9890af (RE-ARC) — LLM-generated grid_maker
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
    bgc, sqc = random.sample(range(10), 2)
    return {"bgc": bgc, "sqc": sqc}


def _unif(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + int(round(lb * (b - a)))
    hi = a + int(round(ub * (b - a)))
    return random.randint(min(lo, hi), max(lo, hi))


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, sqc=1, **kw) -> dict:
    max_h = min(30, max_h)
    max_w = min(30, max_w)
    oh = _unif(diff_lb, diff_ub, (2, max(2, min(5, (max_h - 2) // 2))))
    ow = _unif(diff_lb, diff_ub, (2, max(2, min(5, (max_w - 2) // 2))))
    h = _unif(diff_lb, diff_ub, (2 * oh + 2, max_h))
    w = _unif(diff_lb, diff_ub, (2 * ow + 2, max_w))
    bounds = {(i, j) for i in range(oh) for j in range(ow)}
    obj = {random.choice(sorted(bounds))}

    def shp(s):
        rs = [a for a, _ in s]
        cs = [b for _, b in s]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    while shp(obj) != (oh, ow):
        nb = set()
        for (a, b) in obj:
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    if da or db:
                        nb.add((a + da, b + db))
        obj.add(random.choice(sorted((bounds - obj) & nb)))
    maxfac = 1
    while oh * maxfac + 2 <= h - oh and ow * maxfac + 2 <= w - ow:
        maxfac += 1
    maxfac = max(1, maxfac - 1)
    fac = _unif(diff_lb, diff_ub, (1, maxfac))
    remcols = [c for c in range(10) if c not in (bgc, sqc)]
    numc = _unif(diff_lb, diff_ub, (1, 8))
    ccols = random.sample(remcols, numc)
    objc = {ij: random.choice(ccols) for ij in obj}
    gi = np.full((h, w), bgc, dtype=int)
    SH, SW = oh * fac + 2, ow * fac + 2
    li = random.randint(0, h - SH - oh)
    lj = random.randint(0, w - SW)
    gi[li, lj:lj + SW] = sqc
    gi[li + SH - 1, lj:lj + SW] = sqc
    gi[li:li + SH, lj] = sqc
    gi[li:li + SH, lj + SW - 1] = sqc
    pi = random.randint(li + SH, h - oh)
    pj = random.randint(0, w - ow)
    for (a, b), c in objc.items():
        gi[pi + a, pj + b] = c
    go = np.full((SH, SW), bgc, dtype=int)
    go[0, :] = sqc
    go[-1, :] = sqc
    go[:, 0] = sqc
    go[:, -1] = sqc
    for (a, b), c in objc.items():
        go[1 + a * fac:1 + (a + 1) * fac, 1 + b * fac:1 + (b + 1) * fac] = c
    k = random.randint(0, 3)
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # same-colour 8-connected components
    seen = np.zeros((hi, wi), dtype=bool)
    comps = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or seen[r, c]:
                continue
            col = int(I[r, c])
            seen[r, c] = True
            st = [(r, c)]
            cells = []
            while st:
                y, x = st.pop()
                cells.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == col:
                            seen[ny, nx] = True
                            st.append((ny, nx))
            comps.append(cells)

    # the frame: a component equal to the outline of its bbox (largest)
    best = None
    for cells in comps:
        rs = [y for y, _ in cells]
        cs = [x for _, x in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        if bh < 3 or bw < 3:
            continue
        border = {(y, x) for y in range(r0, r1 + 1) for x in range(c0, c1 + 1)
                  if y in (r0, r1) or x in (c0, c1)}
        if set(cells) == border and (best is None or bh * bw > best[0]):
            best = (bh * bw, r0, c0, bh, bw, border)
    _, br, bc, bh, bw, bcells = best

    # the small pattern: all other non-background cells
    pts = [(r, c) for r in range(hi) for c in range(wi)
           if I[r, c] != bgc and (r, c) not in bcells]
    pr = min(r for r, _ in pts)
    pc = min(c for _, c in pts)
    ph = max(r for r, _ in pts) - pr + 1
    pw = max(c for _, c in pts) - pc + 1
    fy = (bh - 2) // ph
    fx = (bw - 2) // pw

    ops, sels = [], []
    # Row-wise upscaling: stretch each pattern row horizontally into the first
    # interior row of its band, then duplicate that stretched row down the band.
    for i in range(ph):
        top = br + 1 + i * fy
        row_has_copyable = False
        for j in range(pw):
            v = int(I[pr + i, pc + j])
            if v == bgc:
                continue
            c0 = bc + 1 + j * fx
            if v == 0:
                # 0 is invisible to CopyO/Paste: paint its whole block directly
                cells = [(top + a, c0 + b) for a in range(fy) for b in range(fx)]
            else:
                row_has_copyable = True
                cells = [(top, c0 + b) for b in range(fx)]
            ops.append(v)
            sels.append(sel_of(cells))
        if fy > 1 and row_has_copyable:
            # the whole stretched interior row of this band (a full 1-row rectangle)
            ops.append(29)
            sels.append([top, bc + 1, 0, bw - 3])
            for a in range(1, fy):
                ops.append(30)
                sels.append([top + a, bc + 1, 0, 0])

    # keep only the frame and its filled interior (full rectangle crop)
    ops.append(33)
    sels.append([br, bc, bh - 1, bw - 1])
    ops.append(34)
    sels.append([0, 0, bh - 1, bw - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 6b9890af"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 6b9890af"
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
                                f"for task 6b9890af"
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
                    f"Failed to build a complete episode for task 6b9890af "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"6b9890af-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
