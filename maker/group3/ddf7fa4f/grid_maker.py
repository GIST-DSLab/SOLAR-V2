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
from collections import Counter, deque
from maker.sel_helpers import sel_of

ROTS = [0, 1, 2, 3]   # which border line the markers end up on (identity, rot90, rot180, rot270)


def sample_colors(num_examples=None) -> dict:
    bgc = random.randrange(10)
    n = num_examples if num_examples else 3
    ex = list(ROTS) if n >= len(ROTS) else random.sample(ROTS, n)
    ex += [random.choice(ROTS) for _ in range(n - len(ex))]
    random.shuffle(ex)
    plan = [{"rot": r} for r in ex] + [{"rot": random.choice(ex)}]
    return {"bgc": bgc, "instance_plan": plan}


def _analyze(I):
    """bgc, blocks, and the block->marker assignment, read from I alone."""
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    border = list(I[0]) + list(I[-1]) + list(I[:, 0]) + list(I[:, -1])
    bgc = Counter(int(v) for v in border).most_common(1)[0][0]

    # mono-coloured 4-connected components
    seen = np.zeros((h, w), bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or I[r, c] == bgc:
                continue
            col = int(I[r, c])
            comp, dq = [], deque([(r, c)])
            seen[r, c] = True
            while dq:
                a, b = dq.popleft()
                comp.append((a, b))
                for na, nb in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)):
                    if 0 <= na < h and 0 <= nb < w and not seen[na, nb] and I[na, nb] == col:
                        seen[na, nb] = True
                        dq.append((na, nb))
            comps.append((col, comp))

    # marker line = the border line whose touching components all lie entirely ON it
    lines = [{(0, c) for c in range(w)}, {(r, 0) for r in range(h)},
             {(r, w - 1) for r in range(h)}, {(h - 1, c) for c in range(w)}]
    best, bn = None, -1
    for L in lines:
        touching = [cp for cp in comps if any(p in L for p in cp[1])]
        if not touching or not all(all(p in L for p in cp[1]) for cp in touching):
            continue
        n = sum(len(cp[1]) for cp in touching)
        if n > bn:
            bn, best = n, L
    if best is None:
        best = lines[0]

    markers = [(r, c, int(I[r, c])) for (r, c) in sorted(best) if I[r, c] != bgc]
    blocks = [{"color": col, "cells": cp,
               "r0": min(p[0] for p in cp), "r1": max(p[0] for p in cp),
               "c0": min(p[1] for p in cp), "c1": max(p[1] for p in cp)}
              for col, cp in comps if not any(p in best for p in cp)]

    # iterative unique row/column matching
    rb, rm, asg = list(blocks), list(markers), []
    for _ in range(10):
        hits = []
        for b in rb:
            hs = [m for m in rm if b["r0"] <= m[0] <= b["r1"] or b["c0"] <= m[1] <= b["c1"]]
            if len(hs) == 1:
                hits.append((b, hs[0]))
        if not hits:
            break
        for b, m in hits:
            asg.append((b, m))
            rb.remove(b)
            if m in rm:
                rm.remove(m)
    return bgc, blocks, asg


def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, rot=None, **kw):
    if rot is None:
        rot = random.choice(ROTS)

    def unif(lo, hi):
        d = random.uniform(diff_lb, diff_ub)
        return min(hi, max(lo, lo + round(d * (hi - lo))))

    while True:
        hmax = max_h if rot % 2 == 0 else max_w
        wmax = max_w if rot % 2 == 0 else max_h
        h = unif(10, hmax)
        w = unif(10, wmax)
        nocc = unif(1, min(w // 3, (h * w) // 36))
        rem = [c for c in range(10) if c != bgc]
        gi = np.full((h, w), bgc)
        go = gi.copy()
        inds = {(i, j) for i in range(h) for j in range(w) if i > 1}
        succ = tr = 0
        while succ < nocc and tr < 10 * nocc:
            tr += 1
            oh, ow = random.randint(2, 7), random.randint(2, 7)
            free = {j for j in range(w) if gi[0, j] == bgc}
            jopts = {j for j in range(w - ow + 1) if all(jj in free for jj in range(j, j + ow))}
            cands = [(i, j) for (i, j) in inds if i <= h - oh and j in jopts]
            if not cands:
                continue
            li, lj = random.choice(sorted(cands))
            locat = random.choice(range(lj, lj + ow))
            sq = {(i, j) for i in range(li, li + oh) for j in range(lj, lj + ow)}
            if sq <= inds:
                succ += 1
                nb = {(i + a, j + b) for i, j in sq for a, b in ((0, 1), (1, 0), (0, -1), (-1, 0))}
                inds = inds - sq - nb
                col = random.choice(rem)
                gr = random.choice([c for c in rem if c != col])
                gi[0, locat] = col
                go[0, locat] = col
                for i, j in sq:
                    gi[i, j] = gr
                    go[i, j] = col
        gi = np.rot90(gi, rot)
        go = np.rot90(go, rot)
        if gi.shape[0] > max_h or gi.shape[1] > max_w:
            continue
        _, blocks, asg = _analyze(gi)
        if succ == 0 or len(asg) != len(blocks):
            continue          # keep only instances where every block is unambiguously keyed
        return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    bgc, blocks, asg = _analyze(I)

    # Unit of work = one TARGET COLOUR (not one block): every block keyed to a
    # marker of colour k is repainted together by a single Color(k) whose mask is
    # the union of those blocks' own cells (connected objects named from I).
    by_col = {}
    for b, m in asg:
        if m[2] != b["color"]:
            by_col.setdefault(m[2], []).extend(b["cells"])

    ops, sels = [], []
    for k in sorted(by_col):
        ops.append(k)
        sels.append(sel_of(sorted(by_col[k])))

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
