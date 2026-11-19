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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return min(b, max(a, int(round(a + (b - a) * d))))


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)
    max_h = max(10, min(30, max_h))
    max_w = max(10, min(30, max_w))
    h = _unifint(diff_lb, diff_ub, (10, max_h))
    w = _unifint(diff_lb, diff_ub, (10, max_w))
    nobjs = _unifint(diff_lb, diff_ub, (1, 9))
    remcols = [c for c in cols if c != bgc]
    ccols = random.sample(remcols, nobjs)
    gi = [[bgc] * w for _ in range(h)]
    lmocc = set()
    inds = {(i, j) for i in range(h) for j in range(w)}
    succ = 0
    tr = 0
    maxtr = 4 * nobjs
    seenobjs = set()
    mxncells = random.randint(nobjs + 1, 30)

    def dn(c):
        i, j = c
        return [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]

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
                nb.update(dn(c))
            cand = sorted((bounds - obj) & nb)
            obj.add(random.choice(cand))
        mi = min(i for i, j in obj)
        mj = min(j for i, j in obj)
        obj = frozenset((i - mi, j - mj) for i, j in obj)
        if obj in seenobjs:
            continue
        seenobjs.add(obj)
        oh = max(i for i, j in obj) + 1
        ow = max(j for i, j in obj) + 1
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow and ij[1] not in lmocc]
        if not cands:
            continue
        loc = random.choice(sorted(cands))
        plcd = {(i + loc[0], j + loc[1]) for i, j in obj}
        if plcd.issubset(inds):
            nb = set()
            for c in plcd:
                nb.update(dn(c))
            inds = (inds - plcd) - nb
            for i, j in plcd:
                gi[i][j] = ccols[succ]
            succ += 1
            lmocc.add(loc[1])

    comps = _components(np.array(gi), bgc)
    mx = max(len(c[1]) for c in comps)
    big = sorted([c for c in comps if len(c[1]) == mx], key=lambda c: min(j for i, j in c[1]))
    go = [[c[0] for c in big] for _ in range(mx)]
    return {"input": gi, "output": go}


def _components(G, bgc):
    G = np.asarray(G, dtype=int)
    hh, ww = G.shape
    seen = np.zeros((hh, ww), bool)
    comps = []
    for r in range(hh):
        for c in range(ww):
            if seen[r, c] or G[r, c] == bgc:
                continue
            col = G[r, c]
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                a, b = stack.pop()
                cells.append((a, b))
                for na, nb in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)):
                    if 0 <= na < hh and 0 <= nb < ww and not seen[na, nb] and G[na, nb] == col:
                        seen[na, nb] = True
                        stack.append((na, nb))
            comps.append((int(col), cells))
    return comps


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: shared by every input of the episode -> pool counts
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex in examples:
            ei = ex[0] if not isinstance(ex, dict) else ex["input"]
            cnt.update(np.asarray(ei, dtype=int).flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    comps = _components(I, bgc)
    mx = max(len(c[1]) for c in comps)
    big = sorted([c for c in comps if len(c[1]) == mx],
                 key=lambda c: min(j for i, j in c[1]))
    k = len(big)

    ops, sels = [], []
    # Resize canvas to mx rows x k columns (one column per largest object)
    ops.append(33); sels.append([0, 0, mx - 1, k - 1])
    G = np.zeros((mx, k), dtype=int)
    for r in range(min(mx, hi)):
        for c in range(min(k, wi)):
            G[r, c] = I[r, c]

    # Paint each column (full column) with its object's colour, left to right
    for idx, (col, _) in enumerate(big):
        if np.all(G[:, idx] == col):
            continue
        cells = [(r, idx) for r in range(mx)]
        ops.append(col); sels.append(sel_of(cells))
        G[:, idx] = col

    ops.append(34); sels.append([0, 0, mx - 1, k - 1])
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
