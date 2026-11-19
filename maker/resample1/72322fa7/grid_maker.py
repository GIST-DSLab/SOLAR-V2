"""
ARC Task: 72322fa7 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(lb, ub, rng):
    a, b = rng
    lo = a + (b - a) * lb
    hi = a + (b - a) * ub
    return random.randint(int(round(lo)), int(round(max(lo, hi))))


def _nbrs(r, c):
    return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr or dc)]


def sample_colors(num_examples=None) -> dict:
    # Rule is colour-agnostic for the object pairs; only the background is an episode role.
    return {"bgc": random.randint(0, 9)}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    if bgc is None:
        bgc = random.randint(0, 9)
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    remcols = [c for c in range(10) if c != bgc]
    nobjs = _unifint(diff_lb, diff_ub, (1, 4))
    ccols = random.sample(remcols, 2 * nobjs)
    cpairs = list(zip(ccols[:nobjs], ccols[nobjs:]))
    gi = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    objs = []
    for ca, cb in cpairs:
        oh = _unifint(diff_lb, diff_ub, (1, 4))
        ow = _unifint(diff_lb, diff_ub, (2 if oh == 1 else 1, 4))
        if random.choice((True, False)):
            oh, ow = ow, oh
        bounds = {(i, j) for i in range(oh) for j in range(ow)}
        obj = {random.choice(sorted(bounds))}
        ncells = random.randint(2, oh * ow)
        for _ in range(ncells - 1):
            nb = set()
            for (i, j) in obj:
                nb |= set(_nbrs(i, j))
            obj.add(random.choice(sorted((bounds - obj) & nb)))
        mi = min(i for i, j in obj); mj = min(j for i, j in obj)
        objn = sorted((i - mi, j - mj) for i, j in obj)
        aset = set(random.sample(objn, random.randint(1, len(objn) - 1)))
        o = [(ca if c in aset else cb, c) for c in objn]
        cands = [(i, j) for (i, j) in inds if all((i + a, j + b) in inds for a, b in objn)]
        if not cands:
            continue
        loc = random.choice(sorted(cands))
        placed = set()
        for col, (a, b) in o:
            gi[loc[0] + a][loc[1] + b] = col
            placed.add((loc[0] + a, loc[1] + b))
        rem = set(placed)
        for (i, j) in placed:
            rem |= set(_nbrs(i, j))
        inds -= rem
        objs.append(o)
    if not objs:
        raise ValueError("no objects")
    avgs = sum(len(o) for o in objs) / len(objs)
    ub = int(max(1, (h * w) // (avgs * 2)))
    noccs = _unifint(diff_lb, diff_ub, (1, ub))
    succ = 0; tr = 0; maxtr = 5 * noccs
    go = [row[:] for row in gi]
    while tr < maxtr and succ < noccs:
        tr += 1
        o = random.choice(objs)
        oh = max(a for _, (a, b) in o) + 1
        ow = max(b for _, (a, b) in o) + 1
        cands = [(i, j) for (i, j) in inds if i <= h - oh and j <= w - ow]
        if not cands:
            continue
        loc = random.choice(sorted(cands))
        cells = {(loc[0] + a, loc[1] + b) for _, (a, b) in o}
        if cells <= inds:
            succ += 1
            rem = set(cells)
            for (i, j) in cells:
                rem |= set(_nbrs(i, j))
            inds -= rem
            col = random.choice(sorted({c for c, _ in o}))
            for c, (a, b) in o:
                go[loc[0] + a][loc[1] + b] = c
                if c == col:
                    gi[loc[0] + a][loc[1] + b] = c
    # Reject instances where an incidental pattern match makes the rule disagree with go.
    ops, sels = derive_operations(gi, go)
    G = np.asarray(gi, dtype=int).copy()
    for op, sl in zip(ops[:-1], sels[:-1]):
        for (r, c) in sl["cells"]:
            G[r, c] = op
    if not (G == np.asarray(go, dtype=int)).all():
        raise ValueError("ambiguous instance")
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    # background = canvas colour (dominant in this sparse task)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    # complete templates: 8-connected non-bg components containing exactly two colours
    seen = set(); templates = []
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == bgc or (r, c) in seen:
                continue
            comp = []; st = [(r, c)]; seen.add((r, c))
            while st:
                p = st.pop(); comp.append(p)
                for q in _nbrs(*p):
                    if 0 <= q[0] < hi and 0 <= q[1] < wi and q not in seen and I[q] != bgc:
                        seen.add(q); st.append(q)
            if len({int(I[p]) for p in comp}) == 2:
                templates.append(comp)
    G = I.copy()
    ops, sels = [], []
    for comp in templates:
        cols = sorted({int(I[p]) for p in comp})
        for col in cols:
            # one-colour half of the template; find every place it occurs in I
            part = [p for p in comp if I[p] == col]
            ar, ac = min(p[0] for p in part), min(p[1] for p in part)
            pn = [(r - ar, c - ac) for r, c in part]
            full = [(r - ar, c - ac, int(I[r, c])) for r, c in comp]
            ph = max(a for a, b in pn) + 1; pw = max(b for a, b in pn) + 1
            for i in range(hi - ph + 1):
                for j in range(wi - pw + 1):
                    if not all(I[i + a, j + b] == col for a, b in pn):
                        continue
                    # complete the missing other-colour half of this occurrence
                    for pc in cols:
                        if pc == col:
                            continue
                        cells = [(i + a, j + b) for a, b, v in full if v == pc
                                 and 0 <= i + a < hi and 0 <= j + b < wi]
                        cells = [p for p in cells if G[p] != pc]
                        if cells:
                            ops.append(pc); sels.append(sel_of(sorted(cells)))
                            for p in cells:
                                G[p] = pc
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
                        f"num_examples+1 ({num_examples + 1}) for task 72322fa7"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 72322fa7"
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
                                f"for task 72322fa7"
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
                    f"Failed to build a complete episode for task 72322fa7 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"72322fa7-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
