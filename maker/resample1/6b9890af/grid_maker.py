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
    cols = list(range(10))
    bgc, sqc = random.sample(cols, 2)
    remcols = [c for c in cols if c not in (bgc, sqc)]
    numc = random.randint(1, 8)
    ccols = random.sample(remcols, numc)
    return {"bgc": bgc, "sqc": sqc, "ccols": ccols}


def _unif(lb, ub, a, b):
    lo = a + int(round(lb * (b - a)))
    hi = a + int(round(ub * (b - a)))
    lo = max(a, min(lo, b))
    hi = max(lo, min(hi, b))
    return random.randint(lo, hi)


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, sqc=None, ccols=None, **kw) -> dict:
    if bgc is None or sqc is None or ccols is None:
        d = sample_colors()
        bgc, sqc, ccols = d["bgc"], d["sqc"], d["ccols"]
    oh = _unif(diff_lb, diff_ub, 2, 5)
    ow = _unif(diff_lb, diff_ub, 2, 5)
    h = _unif(diff_lb, diff_ub, 2 * oh + 2, max(2 * oh + 2, max_h))
    w = _unif(diff_lb, diff_ub, 2 * ow + 2, max(2 * ow + 2, max_w))
    bounds = {(i, j) for i in range(oh) for j in range(ow)}
    obj = {random.choice(sorted(bounds))}

    def shp(s):
        rs = [a for a, _ in s]
        cs = [b for _, b in s]
        return (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    while shp(obj) != (oh, ow):
        cand = set()
        for (a, b) in obj:
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    if da or db:
                        cand.add((a + da, b + db))
        cand = sorted((cand & bounds) - obj)
        obj.add(random.choice(cand))
    maxfac = 1
    while oh * maxfac + 2 <= h - oh and ow * maxfac + 2 <= w - ow:
        maxfac += 1
    maxfac -= 1
    fac = _unif(diff_lb, diff_ub, 1, max(1, maxfac))
    objc = {ij: random.choice(ccols) for ij in obj}
    gi = np.full((h, w), bgc, dtype=int)
    sh, sw = oh * fac + 2, ow * fac + 2
    li = random.randint(0, h - sh - oh)
    lj = random.randint(0, w - sw)
    gi[li, lj:lj + sw] = sqc
    gi[li + sh - 1, lj:lj + sw] = sqc
    gi[li:li + sh, lj] = sqc
    gi[li:li + sh, lj + sw - 1] = sqc
    li2 = random.randint(li + sh, h - oh)
    lj2 = random.randint(0, w - ow)
    for (a, b), col in objc.items():
        gi[li2 + a, lj2 + b] = col
    go = np.full((sh, sw), bgc, dtype=int)
    go[0, :] = sqc
    go[-1, :] = sqc
    go[:, 0] = sqc
    go[:, -1] = sqc
    for (a, b), col in objc.items():
        go[1 + a * fac:1 + (a + 1) * fac, 1 + b * fac:1 + (b + 1) * fac] = col
    k = random.choice([0, 1, 2, 3])
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # find the largest single-colour 8-connected component that is exactly a rectangle outline (the frame)
    seen = np.zeros_like(I, dtype=bool)
    best = None
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = I[r, c]
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                a, b = stack.pop()
                comp.append((a, b))
                for da in (-1, 0, 1):
                    for db in (-1, 0, 1):
                        x, y = a + da, b + db
                        if 0 <= x < hi and 0 <= y < wi and not seen[x, y] and I[x, y] == col:
                            seen[x, y] = True
                            stack.append((x, y))
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            box = {(x, y) for x in range(r0, r1 + 1) for y in range(c0, c1 + 1)
                   if x in (r0, r1) or y in (c0, c1)}
            if set(comp) == box:
                area = (r1 - r0 + 1) * (c1 - c0 + 1)
                if best is None or area > best[0]:
                    best = (area, r0, c0, r1, c1, set(comp))
    _, fr0, fc0, fr1, fc1, fcells = best
    fh, fw = fr1 - fr0 + 1, fc1 - fc0 + 1

    # the multicolour object = every other non-background cell
    objcells = [(r, c) for r in range(hi) for c in range(wi)
                if I[r, c] != bgc and (r, c) not in fcells]
    or0 = min(r for r, _ in objcells)
    or1 = max(r for r, _ in objcells)
    oc0 = min(c for _, c in objcells)
    oc1 = max(c for _, c in objcells)
    fac_r = (fh - 2) // (or1 - or0 + 1)
    fac_c = (fw - 2) // (oc1 - oc0 + 1)

    ops, sels = [], []
    # crop canvas to the frame: the intended region IS the full frame bbox rectangle
    ops.append(33)
    sels.append([fr0, fc0, fh - 1, fw - 1])
    # paint each object cell as a fac x fac block inside the frame, one op per colour
    bycol = {}
    for (r, c) in objcells:
        bycol.setdefault(int(I[r, c]), []).append((r, c))
    for col in sorted(bycol):
        cells = []
        for (r, c) in sorted(bycol[col]):
            br = 1 + (r - or0) * fac_r
            bc = 1 + (c - oc0) * fac_c
            cells += [(br + i, bc + j) for i in range(fac_r) for j in range(fac_c)]
        ops.append(col)
        sels.append(sel_of(sorted(cells)))
    ops.append(34)
    sels.append([0, 0, fh - 1, fw - 1])
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
