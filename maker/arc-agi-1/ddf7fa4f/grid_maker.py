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


# ---------------------------------------------------------------- 1. colors
def sample_colors(num_examples=None) -> dict:
    """Only bgc is rule-relevant (block/marker colours are arbitrary pairings).
    The rotation applied at the end IS structural: it decides which border
    line carries the single-cell markers, so it is planned per instance."""
    cols = list(range(10))
    bgc = random.choice(cols)

    variants = [{"rot": 0}, {"rot": 1}, {"rot": 2}, {"rot": 3}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(variants):
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(variants, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


# ------------------------------------------------------------- 2. generator
def generate(diff_lb, diff_ub, max_h, max_w, bgc, rot=None) -> dict:
    if rot is None:
        rot = random.choice([0, 1, 2, 3])

    cols = interval(0, 10, 1)

    hcap = max(10, max_h)
    wcap = max(10, max_w)
    if rot in (1, 3):          # rot90 / rot270 swap the axes
        hcap, wcap = wcap, hcap

    h = unifint(diff_lb, diff_ub, (10, hcap))
    w = unifint(diff_lb, diff_ub, (10, wcap))
    nocc = unifint(diff_lb, diff_ub, (1, max(1, min(w // 3, (h * w) // 36))))

    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    succ = 0
    tr = 0
    maxtr = 10 * nocc
    inds = asindices(gi)
    inds = sfilter(inds, lambda ij: ij[0] > 1)
    while succ < nocc and tr < maxtr:
        tr += 1
        oh = randint(2, 7)
        ow = randint(2, 7)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        hastobein = {cidx for cidx, col in enumerate(gi[0]) if col == bgc}
        cantbein = {cidx for cidx, col in enumerate(gi[0]) if col != bgc}
        jopts = [j for j in range(w) if
                 len(set(interval(j, j + ow, 1)) & hastobein) > 0 and
                 len(set(interval(j, j + ow, 1)) & cantbein) == 0]
        cands = sfilter(cands, lambda ij: ij[1] in jopts)
        if len(cands) == 0:
            continue
        loci, locj = choice(totuple(cands))
        locat = choice(sfilter(interval(locj, locj + ow, 1), lambda jj: jj in hastobein))
        sq = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        if sq.issubset(inds):
            succ += 1
            inds = (inds - sq) - mapply(dneighbors, sq)
            col = choice(remcols)
            gr = choice(remove(col, remcols))
            gi = fill(gi, col, {(0, locat)})
            go = fill(go, col, {(0, locat)})
            gi = fill(gi, gr, sq)
            go = fill(go, col, sq)

    rotf = [identity, rot90, rot180, rot270][rot]
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# ------------------------------------------------------ 3. derive operations
def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # --- background = most common colour on the grid's border box -----------
    border = ([int(I[0, c]) for c in range(w)] + [int(I[h - 1, c]) for c in range(w)] +
              [int(I[r, 0]) for r in range(h)] + [int(I[r, w - 1]) for r in range(h)])
    bgc = Counter(border).most_common(1)[0][0]

    # --- the four candidate border lines, in the verifier's order -----------
    lines = [
        [(0, c) for c in range(w)],            # top row
        [(r, 0) for r in range(h)],            # left column
        [(r, w - 1) for r in range(h)],        # right column
        [(h - 1, c) for c in range(w)],        # bottom row
    ]

    def singleton_runs(line):
        """number of length-1 runs of a non-bgc colour along the line"""
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

    # --- markers = the non-bgc cells sitting on that line -------------------
    markers = [(r, c, int(I[r, c])) for (r, c) in best if int(I[r, c]) != bgc]

    # --- blocks = 4-connected mono-coloured components not on the line ------
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
                continue                              # this component IS a marker
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            blocks.append({"cells": comp, "color": col,
                           "r0": min(rs), "r1": max(rs),
                           "c0": min(cs), "c1": max(cs)})

    # --- iterative matching: a block resolves when exactly ONE remaining ----
    # --- marker shares a row or a column with it ---------------------------
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

    # --- emit one recolour per resolved block ------------------------------
    ops, sels = [], []
    for b, m in assignments:
        new_col = m[2]
        if new_col == b["color"]:
            continue
        cellset = set(b["cells"])
        # FloodFill is safe only if no same-coloured cell touches the block
        # (incl. diagonally); otherwise use an exact Color mask.
        contaminated = False
        for (r, c) in b["cells"]:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cellset \
                            and int(I[nr, nc]) == b["color"]:
                        contaminated = True
        if contaminated:
            ops.append(new_col)
            sels.append(sel_of(b["cells"]))
        else:
            ops.append(10 + new_col)                    # FloodFill<new_col>
            sels.append([b["r0"], b["c0"], 0, 0])       # single seed cell

    ops.append(34)
    sels.append([0, 0, O.shape[0] - 1, O.shape[1] - 1])
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
                # backwards-compatible single-key form; v3 uses kwargs dict entries.
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
