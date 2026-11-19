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
from collections import deque
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c != 5]
    bgc, linc = random.sample(cols, 2)
    return {"bgc": bgc, "linc": linc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None) -> dict:
    from dsl import (canvas, backdrop, outbox, box, ulcorner, lrcorner, decrement,
                     increment, asindices, sfilter, fill)
    from random import randint
    from utils import unifint
    if bgc is None or linc is None:
        cols = [c for c in range(10) if c != 5]
        bgc, linc = random.sample(cols, 2)
    h = unifint(diff_lb, diff_ub, (8, max_h))
    w = unifint(diff_lb, diff_ub, (8, max_w))
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
    return {"input": gi, "output": go}


def _ring_sides(r0, c0, r1, c1, h, w):
    """Four straight sides of box (r0,c0)-(r1,c1), each clipped to the grid."""
    sides = []
    cs = range(max(c0, 0), min(c1, w - 1) + 1)
    rs = range(max(r0, 0), min(r1, h - 1) + 1)
    if 0 <= r0 < h:
        sides.append([(r0, c) for c in cs])
    if 0 <= r1 < h and r1 != r0:
        sides.append([(r1, c) for c in cs])
    if 0 <= c0 < w:
        sides.append([(r, c0) for r in rs])
    if 0 <= c1 < w and c1 != c0:
        sides.append([(r, c1) for r in rs])
    return [s for s in sides if s]


def _rings(r0, c0, r1, c1, hoffs, woffs, h, w):
    out = []
    k = 0
    while True:
        sides = _ring_sides(r0 - hoffs * k, c0 - woffs * k,
                            r1 + hoffs * k, c1 + woffs * k, h, w)
        if not sides:
            break
        out.append(sides)
        k += 1
    return out


def _fit(I, linc, bgc):
    """Find the box family (innermost ring + spacings) that redraws I exactly."""
    h, w = I.shape
    L = (I == linc)
    for r0 in range(-4, h):
        for oh in (1, 2, 3):
            r1 = r0 + oh + 1
            for c0 in range(-4, w):
                for ow in (1, 2, 3):
                    c1 = c0 + ow + 1
                    s0 = _ring_sides(r0, c0, r1, c1, h, w)
                    if not s0 or not all(L[r, c] for s in s0 for r, c in s):
                        continue
                    ri = slice(max(r0 + 1, 0), min(r1, h))
                    ci = slice(max(c0 + 1, 0), min(c1, w))
                    if L[ri, ci].any():
                        continue
                    for hoffs in (2, 3, 4):
                        for woffs in (2, 3, 4):
                            rings = _rings(r0, c0, r1, c1, hoffs, woffs, h, w)
                            if len(rings) < 2:
                                continue
                            canvas = np.full((h, w), bgc)
                            npres = 0
                            for sides in rings:
                                if all(L[r, c] for s in sides for r, c in s):
                                    for s in sides:
                                        for r, c in s:
                                            canvas[r, c] = linc
                                    npres += 1
                                else:
                                    break
                            if npres >= 2 and np.array_equal(canvas, I):
                                return rings, npres
    return None


def _components(G, color):
    h, w = G.shape
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if G[r, c] == color and not seen[r, c]:
                q = deque([(r, c)]); seen[r, c] = True; comp = [(r, c)]
                while q:
                    a, b = q.popleft()
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        x, y = a + da, b + db
                        if 0 <= x < h and 0 <= y < w and not seen[x, y] and G[x, y] == color:
                            seen[x, y] = True; q.append((x, y)); comp.append((x, y))
                comps.append(comp)
    return comps


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # bgc / linc: the two colours of I; the box-family fit decides which draws the boxes
    colors = sorted(set(I.flatten().tolist()), key=lambda c: -int((I == c).sum()))
    fit = None
    bgc = linc = None
    for cb in colors:
        for cl in colors:
            if cl == cb:
                continue
            fit = _fit(I, cl, cb)
            if fit is not None:
                bgc, linc = cb, cl
                break
        if fit is not None:
            break
    if fit is None:
        raise ValueError("no box family fits I")
    rings, npresent = fit

    G = I.copy()
    # 1) continue the box family outward: ring by ring (innermost missing first),
    #    each ring drawn one straight side (row / column segment) at a time
    for k in range(npresent, len(rings)):
        for side in rings[k]:
            cells = [(r, c) for r, c in side if G[r, c] != linc]
            if not cells:
                continue
            ops.append(int(linc)); sels.append(sel_of(cells))
            for r, c in cells:
                G[r, c] = linc

    # 2) every background corridor now separated by the boxes floods to 5,
    #    one connected region at a time (seed = one cell of that region)
    for comp in _components(G, bgc):
        r, c = comp[0]
        ops.append(15); sels.append(sel_of([(r, c)]))
        for a, b in comp:
            G[a, b] = 5

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
