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
    rem = [c for c in cols if c != bgc]
    random.shuffle(rem)
    sqcols = rem[:5]          # pool the rectangles draw from
    noisecols = rem[5:]       # pool the noise pixels draw from (disjoint)
    return {"bgc": bgc, "sqcols": sqcols, "noisecols": noisecols}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, sqcols=None, noisecols=None) -> dict:
    cols = interval(0, 10, 1)
    if bgc is None:
        bgc = choice(cols)
    if sqcols is None or noisecols is None:
        rem = [c for c in cols if c != bgc]
        random.shuffle(rem)
        sqcols, noisecols = rem[:5], rem[5:]

    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))

    nsqcols = unifint(diff_lb, diff_ub, (1, len(sqcols)))
    usesqcols = sample(tuple(sqcols), nsqcols)
    nnoisecols = unifint(diff_lb, diff_ub, (1, len(noisecols)))
    usenoisecols = sample(tuple(noisecols), nnoisecols)

    numsq = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 25)))
    succ = 0
    tr = 0
    maxtr = 5 * numsq
    go = canvas(bgc, (h, w))
    inds = asindices(go)
    while tr < maxtr and succ < numsq:
        tr += 1
        oh = randint(2, 7)
        ow = randint(2, 7)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        loci, locj = loc
        sq = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        if sq.issubset(inds):
            succ += 1
            inds = (inds - sq) - outbox(sq)
            col = choice(totuple(usesqcols))
            go = fill(go, col, sq)

    gi = tuple(e for e in go)
    namt = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 9)))
    cands = asindices(gi)
    for k in range(namt):
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        col = gi[loc[0]][loc[1]]
        torem = neighbors(loc) & ofcolor(gi, col)
        cands = cands - torem
        noisec = choice(totuple(usenoisecols))
        gi = fill(gi, noisec, {loc})

    return {'input': gi, 'output': go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # Background = most common colour of I (exactly what the task's rule uses).
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    def comps(G):
        """4-connected, single-coloured, non-background components of G."""
        seen = np.zeros((h, w), dtype=bool)
        out = []
        for r in range(h):
            for c in range(w):
                if seen[r, c] or G[r, c] == bgc:
                    continue
                col = G[r, c]
                cells = {(r, c)}
                seen[r, c] = True
                stack = [(r, c)]
                while stack:
                    rr, cc = stack.pop()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and G[nr, nc] == col:
                            seen[nr, nc] = True
                            cells.add((nr, nc))
                            stack.append((nr, nc))
                out.append((int(col), cells))
        return out

    # ---- Rule, read off I ------------------------------------------------
    # 1. every object of I smaller than 3 cells is noise -> becomes background
    # 2. every object that survives has its whole bounding box filled with its colour
    g6 = I.copy()
    for col, cells in comps(I):
        if len(cells) < 3:
            for (r, c) in cells:
                g6[r, c] = bgc
    survivors = comps(g6)

    boxes = []
    for col, cells in survivors:
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        boxes.append((min(rs), min(cs), max(rs), max(cs), col))
    boxes.sort()

    # target grid implied by the rule (never read from O)
    F = g6.copy()
    for r0, c0, r1, c1, col in boxes:
        F[r0:r1 + 1, c0:c1 + 1] = col

    ops, sels = [], []
    W = I.copy()

    # ---- Step 1: fill each surviving object's backdrop with its colour ----
    # (this also swallows any noise pixel that sits inside that backdrop)
    for r0, c0, r1, c1, col in boxes:
        if np.all(W[r0:r1 + 1, c0:c1 + 1] == col):
            continue                      # box already solid -> nothing to do
        ops.append(col)
        sels.append([r0, c0, r1 - r0, c1 - c0])
        W[r0:r1 + 1, c0:c1 + 1] = col

    # ---- Step 2: wipe the noise objects that lie outside every backdrop ---
    todo = {(r, c) for r in range(h) for c in range(w) if W[r, c] != F[r, c]}
    done = set()
    for (r, c) in sorted(todo):
        if (r, c) in done:
            continue
        col = W[r, c]
        region = {(r, c)}
        stack = [(r, c)]
        while stack:
            rr, cc = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and W[nr, nc] == col:
                    region.add((nr, nc))
                    stack.append((nr, nc))
        if region <= todo and all(F[x, y] == bgc for x, y in region):
            ops.append(10 + int(bgc))     # FloodFill<bgc> from one seed cell
            sels.append([r, c, 0, 0])
            for x, y in region:
                W[x, y] = bgc
            done |= region
        else:
            ops.append(int(F[r, c]))
            sels.append([r, c, 0, 0])
            W[r, c] = F[r, c]
            done.add((r, c))

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
