"""
ARC Task: a3325580 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + (b - a) * diff_lb
    hi = a + (b - a) * diff_ub
    lo_i, hi_i = int(round(lo)), int(round(hi))
    if hi_i < lo_i:
        hi_i = lo_i
    return random.randint(max(a, min(lo_i, b)), max(a, min(hi_i, b)))


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    if bgc is None:
        bgc = random.choice(range(10))
    hmax = max(10, min(30, max_h))
    wmax = max(10, min(30, max_w))
    h = _unifint(diff_lb, diff_ub, (10, hmax))
    w = _unifint(diff_lb, diff_ub, (10, wmax))
    nobjs = _unifint(diff_lb, diff_ub, (1, 9))
    remcols = [c for c in range(10) if c != bgc]
    ccols = random.sample(remcols, nobjs)
    gi = [[bgc] * w for _ in range(h)]
    lmocc = set()
    inds = {(i, j) for i in range(h) for j in range(w)}
    succ = 0
    tr = 0
    maxtr = 4 * nobjs
    seenobjs = set()
    mxncells = random.randint(nobjs + 1, 30)

    def dn(cell):
        i, j = cell
        return {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}

    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = random.randint(1, 6)
        ow = random.randint(1, 6)
        while oh * ow < mxncells:
            oh = random.randint(1, 6)
            ow = random.randint(1, 6)
        bounds = {(i, j) for i in range(oh) for j in range(ow)}
        ncells = _unifint(diff_lb, diff_ub, (1, min(oh * ow, mxncells)))
        ncells = _unifint(diff_lb, diff_ub, (ncells, min(oh * ow, mxncells)))
        sp = random.choice(sorted(bounds))
        obj = {sp}
        for _ in range(ncells - 1):
            nb = set()
            for c in obj:
                nb |= dn(c)
            cand = sorted((bounds - obj) & nb)
            obj.add(random.choice(cand))
        fo = frozenset(obj)
        if fo in seenobjs:
            continue
        seenobjs.add(fo)
        mi = min(i for i, _ in obj); mj = min(j for _, j in obj)
        obj = {(i - mi, j - mj) for i, j in obj}
        oh = max(i for i, _ in obj) + 1
        ow = max(j for _, j in obj) + 1
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow and ij[1] not in lmocc]
        if not cands:
            continue
        loc = random.choice(sorted(cands))
        plcd = {(i + loc[0], j + loc[1]) for i, j in obj}
        if plcd.issubset(inds):
            nbs = set()
            for c in plcd:
                nbs |= dn(c)
            inds = (inds - plcd) - nbs
            for i, j in plcd:
                gi[i][j] = ccols[succ]
            succ += 1
            lmocc.add(loc[1])
    # output: per colour (each colour is one object) sized by count; largest ones, by leftmost
    G = np.array(gi)
    comps = []
    for col in set(G.flatten().tolist()) - {bgc}:
        cells = np.argwhere(G == col)
        comps.append((len(cells), int(cells[:, 1].min()), col))
    mx = max(c[0] for c in comps)
    big = sorted([c for c in comps if c[0] == mx], key=lambda x: x[1])
    go = [[c[2] for c in big] for _ in range(mx)]
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    """Reading: count every colour's cells; the colours tied for the largest count,
    taken in order of first appearance scanning column by column (leftmost first),
    form the answer: a canvas (largest count) tall and (number of such colours) wide,
    where each colour fills one whole column. Columns are painted directly, one per
    winning colour — no row-stripes and no diagonal mirror."""
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    # colour order by first appearance in column-major scan
    order = []
    for c in range(wi):
        for r in range(hi):
            v = int(I[r, c])
            if v != bgc and v not in order:
                order.append(v)
    mx = max(cnt[v] for v in order)
    winners = [v for v in order if cnt[v] == mx]
    m, n = mx, len(winners)

    ops, sels = [], []
    # Resize canvas to m x n (full rectangle anchored at top-left).
    ops.append(33); sels.append([0, 0, m - 1, n - 1])
    # simulate transparent crop: nonzero I cells inside the region survive
    cur = np.zeros((m, n), dtype=int)
    for r in range(min(m, hi)):
        for c in range(min(n, wi)):
            cur[r, c] = I[r, c]

    # Paint one whole column per winning colour, left to right.
    for j, col in enumerate(winners):
        if np.all(cur[:, j] == col):
            continue
        ops.append(int(col)); sels.append([0, j, m - 1, 0])  # full column rectangle
        cur[:, j] = col

    ops.append(34); sels.append([0, 0, m - 1, n - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task a3325580"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a3325580"
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
                                f"for task a3325580"
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
                    f"Failed to build a complete episode for task a3325580 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a3325580-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
