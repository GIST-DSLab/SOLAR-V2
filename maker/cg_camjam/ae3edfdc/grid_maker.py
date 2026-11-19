"""
ARC Task: ae3edfdc (RE-ARC) — LLM-generated grid_maker
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

def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (1, 2, 3, 7)]
    return {"bgc": random.choice(cols)}

def generate(diff_lb, diff_ub, max_h, max_w, bgc=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        lo = a + round((b - a) * lb); hi = a + round((b - a) * ub)
        return random.randint(min(lo, hi), max(lo, hi))
    def nbrs(p):
        return [(p[0] + a, p[1] + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
    sgn = lambda v: (v > 0) - (v < 0)
    if bgc is None:
        bgc = random.choice([c for c in range(10) if c not in (1, 2, 3, 7)])
    h = unifint(diff_lb, diff_ub, (8, max(8, min(30, max_h))))
    w = unifint(diff_lb, diff_ub, (8, max(8, min(30, max_w))))
    go = [[bgc] * w for _ in range(h)]
    rd = (random.randint(1, h - 2), random.randint(1, w - 2))
    excl = set(nbrs(rd)) | {rd}
    reminds = [(i, j) for i in range(1, h - 1) for j in range(1, w - 1) if (i, j) not in excl]
    bd = random.choice(reminds)
    go[rd[0]][rd[1]] = 2
    go[bd[0]][bd[1]] = 1
    gd = random.sample(nbrs(rd), unifint(diff_lb, diff_ub, (1, 8)))
    od = random.sample(nbrs(bd), unifint(diff_lb, diff_ub, (1, 8)))
    for (i, j) in gd: go[i][j] = 3
    for (i, j) in od: go[i][j] = 7
    mpr = {d: (3, (sgn(d[0] - rd[0]), sgn(d[1] - rd[1]))) for d in gd}
    mpr.update({d: (7, (sgn(d[0] - bd[0]), sgn(d[1] - bd[1]))) for d in od})
    ub = (len(gd) + len(od)) * ((h + w) // 5)
    ndist = unifint(diff_lb, diff_ub, (1, ub))
    gi = [row[:] for row in go]
    for _ in range(ndist):
        options = []
        for loc, (col, dr) in mpr.items():
            ii, jj = loc[0] + dr[0], loc[1] + dr[1]
            if 0 <= ii < h and 0 <= jj < w and gi[ii][jj] == bgc:
                options.append((loc, col, dr))
        if not options:
            break
        loc, col, dr = random.choice(options)
        del mpr[loc]
        nl = (loc[0] + dr[0], loc[1] + dr[1])
        mpr[nl] = (col, dr)
        gi[loc[0]][loc[1]] = bgc
        gi[nl[0]][nl[1]] = col
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}

def derive_operations(I, O, examples=None):
    """Reading: each anchor (2, then 1) gathers its own satellites. For one anchor
    the eight rays are swept in turn (below, left, above, right, then the diagonals
    in the same rotation); the dot lying on a ray is pulled back along that ray until
    it touches the anchor. 3s belong to the 2, 7s to the 1."""
    import numpy as np
    from collections import Counter
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: generator draws it from colours outside {1,2,3,7}
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex_i, _ in examples:
            cnt.update(np.asarray(ex_i, dtype=int).flatten().tolist())
    cands = [c for c in cnt if c not in (1, 2, 3, 7)]
    bgc = max(cands, key=lambda c: cnt[c]) if cands else 0

    sgn = lambda v: (v > 0) - (v < 0)
    RAYS = [(1, 0), (0, -1), (-1, 0), (0, 1), (1, -1), (-1, -1), (-1, 1), (1, 1)]

    # per anchor, per ray: the satellite (if any) lying on that ray
    groups = []
    for anchor_col, sat_col in ((2, 3), (1, 7)):
        p = np.argwhere(I == anchor_col)
        if not len(p):
            continue
        ar, ac = int(p[0][0]), int(p[0][1])
        sats = {}
        for cell in np.argwhere(I == sat_col):
            r, c = int(cell[0]), int(cell[1])
            u = (sgn(r - ar), sgn(c - ac))
            sats[u] = (r, c)
        for u in RAYS:
            if u in sats:
                groups.append({"src": sats[u], "dst": (ar + u[0], ac + u[1]),
                               "col": sat_col})

    frozen = {g["src"] for g in groups if g["src"] == g["dst"]}
    pending = [g for g in groups if g["src"] != g["dst"]]

    # keep the anchor-by-anchor sweep order, but never land on a dot still waiting
    order = []
    while pending:
        occupied = frozen | {g["src"] for g in pending}
        pick = next((g for g in pending if g["dst"] not in occupied - {g["src"]}),
                    pending[0])
        pending.remove(pick)
        order.append(pick)

    ops, sels = [], []
    for g in order:
        (sr, sc), (tr, tc) = g["src"], g["dst"]
        vr, vc = tr - sr, tc - sc
        steps = []
        for i in range(max(abs(vr), abs(vc))):   # a diagonal ray alternates U/D and L/R
            if i < abs(vr):
                steps.append(20 if vr < 0 else 21)
            if i < abs(vc):
                steps.append(23 if vc < 0 else 22)
        for k, mop in enumerate(steps):
            ops.append(mop)
            sels.append(sel_of([(sr, sc)]) if k == 0 else sel_of([]))  # grab once
        if bgc != 0:
            ops.append(bgc)                   # restore canvas under the dot's start
            sels.append(sel_of([(sr, sc)]))

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
                        f"num_examples+1 ({num_examples + 1}) for task ae3edfdc"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ae3edfdc"
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
                                f"for task ae3edfdc"
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
                    f"Failed to build a complete episode for task ae3edfdc "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ae3edfdc-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
