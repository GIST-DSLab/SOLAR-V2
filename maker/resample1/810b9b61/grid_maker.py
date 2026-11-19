"""
ARC Task: 810b9b61 (RE-ARC) — LLM-generated grid_maker
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
    # Output colour 3 is hardcoded and never used as background or object colour.
    # The rule depends only on object shape, so only bgc needs to be fixed per episode.
    cols = [c for c in range(10) if c != 3]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    d = random.uniform(diff_lb, diff_ub)
    return max(a, min(b, a + int(round(d * (b - a)))))


def _normalize(s):
    if not s:
        return s
    mr = min(i for i, _ in s)
    mc = min(j for _, j in s)
    return frozenset((i - mr, j - mc) for i, j in s)


def _shape(s):
    rs = [i for i, _ in s]
    cs = [j for _, j in s]
    return max(rs) - min(rs) + 1, max(cs) - min(cs) + 1


def _box(s):
    rs = [i for i, _ in s]
    cs = [j for _, j in s]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    out = set()
    for i in range(r0, r1 + 1):
        for j in range(c0, c1 + 1):
            if i in (r0, r1) or j in (c0, c1):
                out.add((i, j))
    return frozenset(out)


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    cols = [c for c in range(10) if c != 3]
    if bgc is None:
        bgc = random.choice(cols)
    h = _unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = _unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    remcols = [c for c in cols if c != bgc]
    ncols = _unifint(diff_lb, diff_ub, (1, 6))
    ccols = random.sample(remcols, ncols)
    nobjs = _unifint(diff_lb, diff_ub, (3, (h * w) // 10))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}

    def ident(s):
        return s

    def dmir(s):
        return frozenset((j, i) for i, j in s)

    def cmir(s):
        return frozenset((-j, -i) for i, j in s)

    def vmir(s):
        return frozenset((i, -j) for i, j in s)

    def hmir(s):
        return frozenset((-i, j) for i, j in s)

    mfs = (ident, dmir, cmir, vmir, hmir)
    succ, tr, maxtr = 0, 0, 5 * nobjs
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = random.randint(3, 5)
        ow = random.randint(3, 5)
        if oh > h or ow > w:
            continue
        obj = _box(frozenset({(0, 0), (oh - 1, ow - 1)}))
        for fn in random.sample(mfs, random.choice((1, 2))):
            obj = _normalize(fn(obj))
        oh, ow = _shape(obj)
        cands = [(i, j) for (i, j) in inds if i <= h - oh and j <= w - ow]
        if not cands:
            continue
        li, lj = random.choice(cands)
        plcd = frozenset((i + li, j + lj) for i, j in obj)
        if random.choice((True, False)):
            n = _unifint(diff_lb, diff_ub, (1, len(plcd) - 1))
            inobj = frozenset(random.sample(sorted(plcd), n))
        else:
            inobj = plcd
        if inobj.issubset(inds):
            succ += 1
            nb = set()
            for i, j in inobj:
                nb |= {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}
            inds = (inds - inobj) - nb
            col = random.choice(ccols)
            isbox = _box(inobj) == inobj and min(_shape(inobj)) > 2
            for i, j in inobj:
                gi[i][j] = col
                go[i][j] = 3 if isbox else col
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    # Background: the fixed episode colour, i.e. the most common colour across all
    # the inputs (I plus the demonstration inputs).
    cnt = Counter(I.flatten().tolist())
    if examples:
        for ex in examples:
            ei = np.asarray(ex[0], dtype=int)
            cnt.update(ei.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]

    ops, sels = [], []
    seen = np.zeros_like(I, dtype=bool)
    for r in range(hi):
        for c in range(wi):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = I[r, c]
            comp = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == col:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            cs = frozenset(comp)
            sh = _shape(cs)
            # Keep only hollow rectangular outlines at least 3 cells in each direction.
            if min(sh) > 2 and _box(cs) == cs and col != 3:
                seed = min(cs)
                ops.append(13)  # FloodFill3 over the whole connected outline
                sels.append(sel_of([seed]))

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
                        f"num_examples+1 ({num_examples + 1}) for task 810b9b61"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 810b9b61"
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
                                f"for task 810b9b61"
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
                    f"Failed to build a complete episode for task 810b9b61 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"810b9b61-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
