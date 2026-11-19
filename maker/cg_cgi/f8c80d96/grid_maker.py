"""
ARC Task: f8c80d96 (RE-ARC) — LLM-generated grid_maker
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
from dsl import *
from utils import *


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 5]
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc) -> dict:
    h = unifint(diff_lb, diff_ub, (8, max(8, max_h)))
    w = unifint(diff_lb, diff_ub, (8, max(8, max_w)))
    ow = randint(1, 3 if h > 10 else 2)
    oh = randint(1, 3 if w > 10 else 2)
    loci = randint(-oh + 1, h - 1)
    locj = randint(-ow + 1, w - 1)
    obj = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    gi = canvas(bgc, (h, w))
    go = canvas(5, (h, w))
    ulci, ulcj = decrement(ulcorner(obj))
    lrci, lrcj = increment(lrcorner(obj))
    hoffs = randint(2, 4 if h > 12 else 3)
    woffs = randint(2, 4 if w > 12 else 3)
    lns = []
    for k in range(max(h, w) // min(hoffs, woffs) + 1):
        lnx = box(frozenset({(ulci - hoffs * k, ulcj - woffs * k),
                             (lrci + hoffs * k, lrcj + woffs * k)}))
        lns.append(lnx)
    inds = asindices(gi)
    lns = sfilter(lns, lambda ln: len(ln & inds) > 0)
    nlns = len(lns)
    nmissing = unifint(diff_lb, diff_ub, (0, nlns - 2))
    npresent = nlns - nmissing
    for k in range(npresent):
        gi = fill(gi, linc, lns[k])
    for ln in lns:
        go = fill(go, linc, ln)
    return {'input': gi, 'output': go}


def _ring_sides(t, l, b, r, h, w):
    """Four straight sides of box (t,l)-(b,r), clipped to the grid:
    top, right, bottom, left (corners belong to top/bottom)."""
    def clip(cells):
        return [(i, j) for i, j in cells if 0 <= i < h and 0 <= j < w]
    top = clip([(t, j) for j in range(l, r + 1)])
    right = clip([(i, r) for i in range(t + 1, b)])
    bottom = clip([(b, j) for j in range(l, r + 1)])
    left = clip([(i, l) for i in range(t + 1, b)])
    return [top, right, bottom, left]


def _fit(I, linc):
    """Find the box family (innermost box corners + row/col spacing) whose first
    npresent boxes are exactly the linc cells of I.
    Returns (rings as side lists, npresent) or None."""
    h, w = I.shape
    L = (I == linc)
    nL = int(L.sum())
    if nL == 0:
        return None
    for t in range(-4, h):
        for hh in range(3, 6):           # innermost box height = oh + 2
            b = t + hh - 1
            if b < 0:
                continue
            for l in range(-4, w):
                for ww in range(3, 6):
                    r = l + ww - 1
                    if r < 0:
                        continue
                    s0 = [c for s in _ring_sides(t, l, b, r, h, w) for c in s]
                    if not s0 or not all(L[i, j] for i, j in s0):
                        continue
                    for hoffs in (2, 3, 4):
                        for woffs in (2, 3, 4):
                            rings = []
                            for k in range(max(h, w) // min(hoffs, woffs) + 1):
                                sides = _ring_sides(t - hoffs * k, l - woffs * k,
                                                    b + hoffs * k, r + woffs * k, h, w)
                                if any(sides):
                                    rings.append(sides)
                            M = np.zeros_like(L)
                            for n, sides in enumerate(rings):
                                for s in sides:
                                    for i, j in s:
                                        M[i, j] = True
                                if int(M.sum()) >= nL:
                                    if n + 1 >= 2 and np.array_equal(M, L):
                                        return rings, n + 1
                                    break
    return None


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []
    present = sorted(set(I.flatten().tolist()))

    # Background convention read from the demonstrations: the colour that turns 5.
    bgc = None
    if examples:
        for ei, eo in examples:
            ei = np.asarray(ei, dtype=int)
            eo = np.asarray(eo, dtype=int)
            vals = set(ei[eo == 5].tolist())
            if len(vals) == 1:
                bgc = vals.pop()
                break
    cands = []
    if bgc is not None and bgc in present:
        cands = [(c, bgc) for c in present if c != bgc]
    cands += [(a, b) for a in present for b in present
              if a != b and (a, b) not in cands]

    fit = None
    linc = None
    for cl, cb in cands:
        fit = _fit(I, cl)
        if fit is not None:
            linc, bgc = cl, cb
            break
    if fit is None:
        raise ValueError("no box family fits the input")
    rings, npresent = fit

    # 1) Continue the box family outward, innermost missing box first; each box
    #    is drawn as its straight sides (top, right, bottom, left), clipped to grid.
    cur = I.copy()
    for k in range(npresent, len(rings)):
        for side in rings[k]:
            if side:
                ops.append(int(linc))
                sels.append(sel_of(side))
                for i, j in side:
                    cur[i, j] = linc

    # 2) Every cell still showing the old background colour becomes 5.
    bg_cells = [(i, j) for i in range(h) for j in range(w) if cur[i, j] == bgc]
    if bg_cells:
        ops.append(5)
        sels.append(sel_of(bg_cells))

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task f8c80d96"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task f8c80d96"
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
                                f"for task f8c80d96"
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
                    f"Failed to build a complete episode for task f8c80d96 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"f8c80d96-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
