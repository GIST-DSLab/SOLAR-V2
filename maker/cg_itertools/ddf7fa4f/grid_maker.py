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

ROTS = [0, 1, 2, 3]

def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROTS):
        ex = [{"rot": r} for r in ROTS] + [{"rot": random.choice(ROTS)} for _ in range(n_ex - len(ROTS))]
        random.shuffle(ex)
    else:
        ex = [{"rot": r} for r in random.sample(ROTS, n_ex)]
    plan = ex + [dict(random.choice(ex))]
    return {"bgc": bgc, "instance_plan": plan}

def generate(diff_lb, diff_ub, max_h=30, max_w=30, bgc=0, rot=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a: b = a
        d = random.uniform(lb, ub)
        return max(a, min(b, a + round(d * (b - a))))
    if rot is None:
        rot = random.choice(ROTS)
    cols = list(range(10))
    h = unifint(diff_lb, diff_ub, (10, max_h))
    w = unifint(diff_lb, diff_ub, (10, max_w))
    nocc = unifint(diff_lb, diff_ub, (1, max(1, min(w // 3, (h * w) // 36))))
    remcols = [c for c in cols if c != bgc]
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    succ, tr, maxtr = 0, 0, 10 * nocc
    inds = {(i, j) for i in range(h) for j in range(w) if i > 1}
    while succ < nocc and tr < maxtr:
        tr += 1
        oh = random.randint(2, 7); ow = random.randint(2, 7)
        hastobein = {j for j in range(w) if gi[0][j] == bgc}
        cantbein = {j for j in range(w) if gi[0][j] != bgc}
        jopts = {j for j in range(w)
                 if set(range(j, j + ow)) & hastobein and not (set(range(j, j + ow)) & cantbein)}
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow and ij[1] in jopts]
        if not cands:
            continue
        loci, locj = random.choice(sorted(cands))
        locat = random.choice([jj for jj in range(locj, locj + ow) if jj in hastobein])
        sq = {(i, j) for i in range(loci, loci + oh) for j in range(locj, locj + ow)}
        if sq <= inds:
            succ += 1
            nb = set()
            for (i, j) in sq:
                nb |= {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}
            inds = (inds - sq) - nb
            col = random.choice(remcols)
            gr = random.choice([c for c in remcols if c != col])
            gi[0][locat] = col; go[0][locat] = col
            for (i, j) in sq:
                gi[i][j] = gr; go[i][j] = col
    gi = np.rot90(np.array(gi), -rot); go = np.rot90(np.array(go), -rot)
    return {"input": gi.tolist(), "output": go.tolist()}

def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    border = ([int(I[0, c]) for c in range(w)] + [int(I[h - 1, c]) for c in range(w)] +
              [int(I[r, 0]) for r in range(h)] + [int(I[r, w - 1]) for r in range(h)])
    bgc = Counter(border).most_common(1)[0][0]
    sides = {
        "top":    [(0, c) for c in range(w)],
        "left":   [(r, 0) for r in range(h)],
        "right":  [(r, w - 1) for r in range(h)],
        "bottom": [(h - 1, c) for c in range(w)],
    }
    # the marker side: the border whose non-background cells form components that
    # live entirely on that border (markers), not components reaching into the grid
    def comp_of(r0, c0):
        col = int(I[r0, c0]); seen_ = {(r0, c0)}; dq = deque([(r0, c0)])
        while dq:
            a, b = dq.popleft()
            for na, nb in ((a-1, b), (a+1, b), (a, b-1), (a, b+1)):
                if 0 <= na < h and 0 <= nb < w and (na, nb) not in seen_ and I[na, nb] == col:
                    seen_.add((na, nb)); dq.append((na, nb))
        return seen_
    def score(line):
        ls = set(line); good = bad = 0
        for r, c in line:
            if I[r, c] == bgc: continue
            if comp_of(r, c) <= ls: good += 1
            else: bad += 1
        return (good, -bad)
    side, best_n = "top", None
    for k in ("top", "left", "right", "bottom"):
        n = score(sides[k])
        if best_n is None or n > best_n: best_n, side = n, k
    line = sides[side]; line_set = set(line)
    markers = [(r, c, int(I[r, c])) for r, c in line if int(I[r, c]) != bgc]
    vertical_beam = side in ("top", "bottom")
    seen = np.zeros((h, w), bool); blocks = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or I[r, c] == bgc: continue
            col = int(I[r, c]); comp = []; dq = deque([(r, c)]); seen[r, c] = True
            while dq:
                a, b = dq.popleft(); comp.append((a, b))
                for na, nb in ((a-1, b), (a+1, b), (a, b-1), (a, b+1)):
                    if 0 <= na < h and 0 <= nb < w and not seen[na, nb] and I[na, nb] == col:
                        seen[na, nb] = True; dq.append((na, nb))
            if any(p in line_set for p in comp): continue
            rs = [p[0] for p in comp]; cs = [p[1] for p in comp]
            blocks.append({"cells": comp, "color": col, "r0": min(rs), "r1": max(rs),
                           "c0": min(cs), "c1": max(cs)})
    def beam_hits(m, b):
        # the marker's beam runs perpendicular to its border side
        return b["c0"] <= m[1] <= b["c1"] if vertical_beam else b["r0"] <= m[0] <= b["r1"]
    def entry(m, b):
        # first cell of the block the marker's beam enters
        if side == "top": return (b["r0"], m[1])
        if side == "bottom": return (b["r1"], m[1])
        if side == "left": return (m[0], b["c0"])
        return (m[0], b["c1"])
    rem_b, rem_m, ops, sels = list(blocks), list(markers), [], []
    while rem_b and rem_m:
        rnd = []
        for b in rem_b:
            hits = [m for m in rem_m if beam_hits(m, b)]
            if len(hits) == 1: rnd.append((hits[0], b))
        if not rnd: break
        rnd.sort(key=lambda mb: (mb[0][0], mb[0][1]))   # walk markers along the border
        for m, b in rnd:
            rem_b.remove(b)
            if m in rem_m: rem_m.remove(m)
            if m[2] == b["color"]: continue
            cs = set(b["cells"])
            touch = any(0 <= r+dr < h and 0 <= c+dc < w and (r+dr, c+dc) not in cs
                        and I[r+dr, c+dc] == b["color"]
                        for r, c in b["cells"] for dr in (-1, 0, 1) for dc in (-1, 0, 1))
            if touch:
                ops.append(m[2]); sels.append(sel_of(b["cells"]))
            else:
                er, ec = entry(m, b)
                ops.append(10 + m[2]); sels.append([er, ec, 0, 0])
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
