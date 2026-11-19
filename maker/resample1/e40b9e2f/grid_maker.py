"""
ARC Task: e40b9e2f (RE-ARC) — LLM-generated grid_maker
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
from dsl import *
from utils import *
from random import choice, randint, sample


def sample_colors(num_examples=None) -> dict:
    # The rule (complete 4-fold rotational symmetry) is colour-agnostic; only bgc is fixed.
    return {"bgc": random.choice(range(10))}


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=None, **kw) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = random.choice(cols)
    h = unifint(diff_lb, diff_ub, (6, max_h))
    w = unifint(diff_lb, diff_ub, (6, max_w))
    d = unifint(diff_lb, diff_ub, (4, min(h, w) - 2))
    loci = randint(0, h - d)
    locj = randint(0, w - d)
    loc = (loci, locj)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numcols)
    subg = canvas(bgc, (d, d))
    inds = asindices(subg)
    if d % 2 == 0:
        q = sfilter(inds, lambda ij: ij[0] < d // 2 and ij[1] < d // 2)
        cp = {(d//2-1, d//2-1), (d//2, d//2-1), (d//2-1, d//2), (d//2, d//2)}
    else:
        q = sfilter(inds, lambda ij: ij[0] < d // 2 and ij[1] <= d // 2)
        cp = {(d//2, d//2)} | ineighbors((d//2, d//2))
    nrings = unifint(diff_lb, diff_ub, (1, max(1, (d - 2) // 2)))
    rings = set()
    for k in range(nrings):
        ring = box({(k, k), (d - k - 1, d - k - 1)})
        rings = rings | ring
    qin = q - rings
    qout = rings & q

    def mk():
        ntailobjcells = unifint(diff_lb, diff_ub, (1, len(q)))
        tailobjcells = sample(totuple(q), ntailobjcells)
        tailobjcells = set(tailobjcells) | {choice(totuple(qin))} | {choice(totuple(qout))}
        return {(choice(ccols), ij) for ij in tailobjcells}

    tailobj = mk()
    while hmirror(tailobj) == tailobj and vmirror(tailobj) == tailobj:
        tailobj = mk()
    for k in range(4):
        subg = paint(subg, tailobj)
        subg = rot90(subg)
    fxobj = recolor(choice(ccols), cp)
    subg = paint(subg, fxobj)
    subgi = subg
    subgo = tuple(e for e in subgi)
    subgi = fill(subgi, bgc, rings)
    nsplits = unifint(diff_lb, diff_ub, (1, 4))
    splits = [set() for k in range(nsplits)]
    for idx, cel in enumerate(tailobj):
        splits[idx % nsplits].add(cel)
    for jj in range(4):
        if jj < len(splits):
            subgi = paint(subgi, splits[jj])
        subgi = rot90(subgi)
    subgi = paint(subgi, fxobj)
    rotf = choice((identity, rot90, rot180, rot270))
    subgi = rotf(subgi)
    subgo = rotf(subgo)
    gi = paint(canvas(bgc, (h, w)), shift(asobject(subgi), loc))
    go = paint(canvas(bgc, (h, w)), shift(asobject(subgo), loc))
    return {"input": gi, "output": go}


def _bgc(I, examples):
    # background = colour present on all four borders of every demo input
    if examples:
        cands = None
        for ei, _ in examples:
            E = np.asarray(ei, dtype=int)
            border = (set(E[0].tolist()) & set(E[-1].tolist())
                      & set(E[:, 0].tolist()) & set(E[:, -1].tolist()))
            cands = border if cands is None else cands & border
        if cands:
            cnt = Counter(I.flatten().tolist())
            c = max(cands, key=lambda x: cnt.get(x, 0))
            if cnt.get(c, 0) > 0:
                return c
    return Counter(I.flatten().tolist()).most_common(1)[0][0]


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    bgc = _bgc(I, examples)
    fg = {(r, c): int(I[r, c]) for r in range(hi) for c in range(wi) if I[r, c] != bgc}
    rs = [p[0] for p in fg]; cs = [p[1] for p in fg]
    rmin, rmax, cmin, cmax = min(rs), max(rs), min(cs), max(cs)

    def rot(p, a2, b2):  # CW quarter-turn about (a2/2, b2/2) (doubled coordinates)
        r, c = p
        return ((a2 + 2 * c - b2) // 2, (b2 - 2 * r + a2) // 2)

    def consistent(a2, b2):
        score = 0
        for p, col in fg.items():
            q = p
            for _ in range(3):
                q = rot(q, a2, b2)
                if not (0 <= q[0] < hi and 0 <= q[1] < wi):
                    return None
                if q in fg:
                    if fg[q] != col:
                        return None
                    score += 1
        return score

    # Pivot = centre of the largest square window (size 7..2, inside the pattern
    # bbox) that is already fully 4-fold symmetric in I (the central core);
    # ties go to the window nearest the bbox centre.
    hb, wb = rmax - rmin + 1, cmax - cmin + 1
    ctr = (rmin + hb // 2, cmin + wb // 2)
    cands = []
    for s in range(7, 1, -1):
        for i in range(rmin, rmin + hb - s + 1):
            for j in range(cmin, cmin + wb - s + 1):
                W = I[i:i + s, j:j + s]
                if W.shape != (s, s):
                    continue
                if not (np.array_equal(W, np.rot90(W)) and np.array_equal(W, np.rot90(W, 2))):
                    continue
                if int((W == bgc).sum()) >= s * s - 1:
                    continue
                dist = abs(i + s // 2 - ctr[0]) + abs(j + s // 2 - ctr[1])
                cands.append((-s, dist, 2 * i + s - 1, 2 * j + s - 1))
    cands.sort()
    pick = None
    for _, _, a2, b2 in cands:
        if consistent(a2, b2) is not None:
            pick = (a2, b2)
            break
    if pick is None:
        pick = (2 * ctr[0], 2 * ctr[1])
    a2, b2 = pick

    G = I.copy()
    ops, sels = [], []
    # For each quarter-turn k, stamp the k-times-rotated copy of I's pattern,
    # one Color op per colour, only on cells that currently differ.
    for k in (1, 2, 3):
        by_col = {}
        for p, col in fg.items():
            q = p
            for _ in range(k):
                q = rot(q, a2, b2)
            if not (0 <= q[0] < hi and 0 <= q[1] < wi):
                continue
            if G[q] != col:
                by_col.setdefault(col, []).append(q)
        for col in sorted(by_col):
            cells = sorted(set(by_col[col]))
            ops.append(col); sels.append(sel_of(cells))
            for q in cells:
                G[q] = col
    ops.append(34); sels.append(sel_of([(r, c) for r in range(hi) for c in range(wi)]))
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
                        f"num_examples+1 ({num_examples + 1}) for task e40b9e2f"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e40b9e2f"
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
                                f"for task e40b9e2f"
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
                    f"Failed to build a complete episode for task e40b9e2f "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e40b9e2f-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
