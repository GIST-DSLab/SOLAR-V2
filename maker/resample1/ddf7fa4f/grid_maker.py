"""
ARC Task: ddf7fa4f (RE-ARC) — LLM-generated grid_maker
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
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def _unifint(lb, ub, rng):
    a, b = rng
    if b < a:
        b = a
    lo = a + (b - a) * min(lb, ub)
    hi = a + (b - a) * max(lb, ub)
    return random.randint(int(round(lo)), max(int(round(lo)), int(round(hi))))


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, **kw) -> dict:
    cols = list(range(10))
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    rotk = random.choice([0, 1, 2, 3])
    if rotk % 2 == 1:
        # after rotation dims swap; keep within bounds
        h = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
        w = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    nocc = _unifint(diff_lb, diff_ub, (1, max(1, min(w // 3, (h * w) // 36))))
    remcols = [c for c in cols if c != bgc]
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    succ = 0
    tr = 0
    maxtr = 10 * nocc
    inds = {(i, j) for i in range(h) for j in range(w) if i > 1}
    while succ < nocc and tr < maxtr:
        tr += 1
        oh = random.randint(2, 7)
        ow = random.randint(2, 7)
        hastobein = {j for j in range(w) if gi[0, j] == bgc}
        cantbein = {j for j in range(w) if gi[0, j] != bgc}
        jopts = set(j for j in range(w)
                    if len(set(range(j, j + ow)) & hastobein) > 0
                    and len(set(range(j, j + ow)) & cantbein) == 0)
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow and ij[1] in jopts]
        if not cands:
            continue
        loci, locj = random.choice(cands)
        locat = random.choice([jj for jj in range(locj, locj + ow) if jj in hastobein])
        sq = {(i, j) for i in range(loci, loci + oh) for j in range(locj, locj + ow)}
        if sq.issubset(inds):
            succ += 1
            nb = set()
            for (i, j) in sq:
                nb |= {(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)}
            inds = (inds - sq) - nb
            col = random.choice(remcols)
            gr = random.choice([c for c in remcols if c != col])
            gi[0, locat] = col
            go[0, locat] = col
            for (i, j) in sq:
                gi[i, j] = gr
                go[i, j] = col
    # np.rot90 k=3 == CW; any rotation equally valid
    gi = np.rot90(gi, rotk)
    go = np.rot90(go, rotk)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    border = [I[0, c] for c in range(wi)] + [I[hi - 1, c] for c in range(wi)] + \
             [I[r, 0] for r in range(hi)] + [I[r, wi - 1] for r in range(hi)]
    bgc = Counter(border).most_common(1)[0][0]

    # candidate marker lines: (cells on line, inward neighbor fn, axis for projection)
    lines = {
        "top": ([(0, c) for c in range(wi)], lambda r, c: (1, c), 1),
        "bottom": ([(hi - 1, c) for c in range(wi)], lambda r, c: (hi - 2, c), 1),
        "left": ([(r, 0) for r in range(hi)], lambda r, c: (r, 1), 0),
        "right": ([(r, wi - 1) for r in range(hi)], lambda r, c: (r, wi - 2), 0),
    }

    def components(excl):
        seen = set()
        comps = []
        for r in range(hi):
            for c in range(wi):
                if (r, c) in seen or (r, c) in excl or I[r, c] == bgc:
                    continue
                col = I[r, c]
                stack = [(r, c)]
                seen.add((r, c))
                cells = []
                while stack:
                    a, b = stack.pop()
                    cells.append((a, b))
                    for na, nb in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                        if 0 <= na < hi and 0 <= nb < wi and (na, nb) not in seen \
                                and (na, nb) not in excl and I[na, nb] == col:
                            seen.add((na, nb))
                            stack.append((na, nb))
                comps.append(cells)
        return comps

    best = None
    for name, (cells, inward, axis) in lines.items():
        markers = {}
        for (r, c) in cells:
            if I[r, c] != bgc:
                ir, ic = inward(r, c)
                if I[ir, ic] == bgc:
                    markers[c if axis == 1 else r] = int(I[r, c])
        if not markers:
            continue
        comps = components(set(cells))
        matched = 0
        plan = []
        for comp in comps:
            coords = [p[axis] for p in comp]
            lo, hi_ = min(coords), max(coords)
            ms = [k for k in markers if lo <= k <= hi_]
            if len(ms) == 1:
                matched += 1
                plan.append((comp, markers[ms[0]]))
        score = (matched == len(comps), matched, len(markers))
        if best is None or score > best[0]:
            best = (score, plan)

    ops, sels = [], []
    if best is not None:
        for comp, mcol in sorted(best[1], key=lambda t: min(t[0])):
            r, c = min(comp)
            if I[r, c] == mcol:
                continue
            ops.append(10 + mcol)
            sels.append([r, c, 0, 0])
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
                        f"num_examples+1 ({num_examples + 1}) for task ddf7fa4f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ddf7fa4f"
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
                                f"for task ddf7fa4f"
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
                    f"Failed to build a complete episode for task ddf7fa4f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ddf7fa4f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
