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
from collections import Counter, deque
import numpy as np
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return max(a, min(b, int(round(random.uniform(lo, hi)))))

    if bgc is None:
        bgc = random.choice(range(10))
    cols = list(range(10))
    max_h = max(10, min(30, max_h))
    max_w = max(10, min(30, max_w))
    rot = random.choice([0, 1, 2, 3])
    # rotation by 90/270 swaps the dimensions; keep the final grid within limits
    if rot in (1, 3):
        hcap, wcap = max_w, max_h
    else:
        hcap, wcap = max_h, max_w
    h = unifint(diff_lb, diff_ub, (10, hcap))
    w = unifint(diff_lb, diff_ub, (10, wcap))
    nocc = unifint(diff_lb, diff_ub, (1, max(1, min(w // 3, (h * w) // 36))))
    remcols = [c for c in cols if c != bgc]
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    succ, tr, maxtr = 0, 0, 10 * nocc
    inds = {(i, j) for i in range(h) for j in range(w) if i > 1}
    while succ < nocc and tr < maxtr:
        tr += 1
        oh = random.randint(2, 7)
        ow = random.randint(2, 7)
        hastobein = {j for j in range(w) if gi[0, j] == bgc}
        cantbein = {j for j in range(w) if gi[0, j] != bgc}
        jopts = [j for j in range(w)
                 if len(set(range(j, j + ow)) & hastobein) > 0
                 and len(set(range(j, j + ow)) & cantbein) == 0]
        jset = set(jopts)
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow and ij[1] in jset]
        if not cands:
            continue
        loci, locj = random.choice(cands)
        locat = random.choice([jj for jj in range(locj, locj + ow) if jj in hastobein])
        sq = {(i, j) for i in range(loci, loci + oh) for j in range(locj, locj + ow)}
        if sq.issubset(inds):
            succ += 1
            nb = set()
            for (i, j) in sq:
                nb |= {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}
            inds = (inds - sq) - nb
            col = random.choice(remcols)
            gr = random.choice([c for c in remcols if c != col])
            gi[0, locat] = col
            go[0, locat] = col
            for (i, j) in sq:
                gi[i, j] = gr
                go[i, j] = col
    # rot90 (CW) = np.rot90(k=3), rot180 = k=2, rot270 = k=1
    k = {0: 0, 1: 3, 2: 2, 3: 1}[rot]
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # background = most common colour on the border box (verifier's definition)
    border = ([int(I[0, c]) for c in range(w)] + [int(I[h - 1, c]) for c in range(w)] +
              [int(I[r, 0]) for r in range(h)] + [int(I[r, w - 1]) for r in range(h)])
    bgc = Counter(border).most_common(1)[0][0]

    # the marker line = border line with the most single-cell non-bgc runs
    lines = [
        [(0, c) for c in range(w)],
        [(r, 0) for r in range(h)],
        [(r, w - 1) for r in range(h)],
        [(h - 1, c) for c in range(w)],
    ]

    def singleton_runs(line):
        n, i = 0, 0
        L = [int(I[r, c]) for (r, c) in line]
        while i < len(L):
            if L[i] == bgc:
                i += 1
                continue
            j = i
            while j < len(L) and L[j] == L[i]:
                j += 1
            if j - i == 1:
                n += 1
            i = j
        return n

    best, best_n = lines[0], -1
    for ln in lines:
        n = singleton_runs(ln)
        if n > best_n:
            best_n, best = n, ln
    line_set = set(best)
    markers = [(r, c, int(I[r, c])) for (r, c) in best if int(I[r, c]) != bgc]

    # blocks = 4-connected single-colour components that don't touch the marker line
    seen = np.zeros((h, w), dtype=bool)
    blocks = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(I[r, c]) == bgc:
                continue
            col = int(I[r, c])
            comp, dq = [], deque([(r, c)])
            seen[r, c] = True
            while dq:
                cr, cc = dq.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(I[nr, nc]) == col:
                        seen[nr, nc] = True
                        dq.append((nr, nc))
            if any(p in line_set for p in comp):
                continue
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            blocks.append({"cells": comp, "color": col,
                           "r0": min(rs), "r1": max(rs), "c0": min(cs), "c1": max(cs)})

    # a block takes its colour from the one remaining marker that shares a row or column with it
    rem_blocks = list(blocks)
    rem_marks = list(markers)
    assignments = []
    for _ in range(10):
        if not rem_blocks or not rem_marks:
            break
        round_hits = []
        for b in rem_blocks:
            hits = [m for m in rem_marks
                    if (b["r0"] <= m[0] <= b["r1"]) or (b["c0"] <= m[1] <= b["c1"])]
            if len(hits) == 1:
                round_hits.append((b, hits[0]))
        if not round_hits:
            break
        round_hits.sort(key=lambda bm: (bm[0]["r0"], bm[0]["c0"]))
        for b, m in round_hits:
            assignments.append((b, m))
            rem_blocks.remove(b)
            if m in rem_marks:
                rem_marks.remove(m)

    # recolour each whole block in one stroke
    ops, sels = [], []
    for b, m in assignments:
        new_col = m[2]
        if new_col == b["color"]:
            continue
        cellset = set(b["cells"])
        contaminated = False
        for (r, c) in b["cells"]:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cellset \
                        and int(I[nr, nc]) == b["color"]:
                    contaminated = True
        if contaminated:
            ops.append(new_col)
            sels.append(sel_of(b["cells"]))           # exact block cells
        else:
            ops.append(10 + new_col)                  # FloodFill the block
            sels.append([b["r0"], b["c0"], 0, 0])

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
