"""
ARC Task: 7e0986d6 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    nsq = random.randint(1, 5)
    sqcols = random.sample(remcols, nsq)
    remcols = [c for c in remcols if c not in sqcols]
    nnoise = random.randint(1, len(remcols))
    noisecols = random.sample(remcols, nnoise)
    return {"bgc": bgc, "sqcols": sqcols, "noisecols": noisecols}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, sqcols, noisecols) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round((b - a) * lb))
        hi = a + int(round((b - a) * ub))
        hi = max(lo, hi)
        return random.randint(lo, hi)

    h = unifint(diff_lb, diff_ub, (10, max(10, max_h)))
    w = unifint(diff_lb, diff_ub, (10, max(10, max_w)))
    numsq = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 25)))
    go = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    succ = 0
    tr = 0
    maxtr = 5 * numsq
    while tr < maxtr and succ < numsq:
        tr += 1
        oh = random.randint(2, 7)
        ow = random.randint(2, 7)
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cands:
            continue
        li, lj = random.choice(sorted(cands))
        sq = {(i, j) for i in range(li, li + oh) for j in range(lj, lj + ow)}
        if sq <= inds:
            succ += 1
            ob = {(i, j) for i in range(li - 1, li + oh + 1) for j in range(lj - 1, lj + ow + 1)}
            inds = inds - ob
            col = random.choice(sqcols)
            for i, j in sq:
                go[i][j] = col
    gi = [row[:] for row in go]
    namt = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 9)))
    cands = {(i, j) for i in range(h) for j in range(w)}
    for _ in range(namt):
        if not cands:
            break
        loc = random.choice(sorted(cands))
        col = gi[loc[0]][loc[1]]
        nb = {(loc[0] + a, loc[1] + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)}
        torem = {p for p in nb if 0 <= p[0] < h and 0 <= p[1] < w and gi[p[0]][p[1]] == col}
        cands -= torem
        gi[loc[0]][loc[1]] = random.choice(noisecols)
    return {"input": [tuple(r) for r in gi], "output": [tuple(r) for r in go]}


def derive_operations(I, O=None, examples=None):
    # Unit of work = colour class over the whole grid, not object by object:
    #   1) ONE Color(bgc) wiping every stray speckle lying outside all blocks
    #      (all noise colours together = "everything off-block that isn't bg"),
    #   2) ONE Color(col) per block colour over the union of all rectangles of
    #      that colour (each rectangle = backdrop of a >=3-cell component of I),
    #      which overwrites the in-block speckles at once.
    # Everything is measured from I; O is never read.
    from maker.sel_helpers import sel_of

    I = np.asarray(I, dtype=int)
    h, w = I.shape

    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            col = int(I[r, c])
            seen[r, c] = True
            st = [(r, c)]
            cells = []
            while st:
                a, b = st.pop()
                cells.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    na, nb = a + da, b + db
                    if 0 <= na < h and 0 <= nb < w and not seen[na, nb] and I[na, nb] == col:
                        seen[na, nb] = True
                        st.append((na, nb))
            comps.append((col, cells))

    counts = Counter(I.flatten().tolist())
    bgc = max(comps, key=lambda t: (len(t[1]), counts[t[0]]))[0]

    rects = {}
    inside = set()
    for col, cells in comps:
        if col == bgc or len(cells) < 3:
            continue
        r0 = min(a for a, b in cells)
        r1 = max(a for a, b in cells)
        c0 = min(b for a, b in cells)
        c1 = max(b for a, b in cells)
        rect = [(a, b) for a in range(r0, r1 + 1) for b in range(c0, c1 + 1)]
        rects.setdefault(col, []).extend(rect)
        inside.update(rect)

    ops, sels = [], []
    cur = I.copy()

    # 1) all off-block non-background cells -> background, in one stroke
    stray = [(r, c) for r in range(h) for c in range(w)
             if (r, c) not in inside and cur[r, c] != bgc]
    if stray:
        ops.append(int(bgc))
        sels.append(sel_of(stray))
        for a, b in stray:
            cur[a, b] = bgc

    # 2) per block colour: fill the union of its full rectangles
    for col in sorted(rects):
        cells = sorted(set(rects[col]))
        if any(cur[a, b] != col for a, b in cells):
            ops.append(int(col))
            sels.append(sel_of(cells))
            for a, b in cells:
                cur[a, b] = col

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
                        f"num_examples+1 ({num_examples + 1}) for task 7e0986d6"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 7e0986d6"
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
                                f"for task 7e0986d6"
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
                    f"Failed to build a complete episode for task 7e0986d6 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"7e0986d6-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
