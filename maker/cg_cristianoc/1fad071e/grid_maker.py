"""
ARC Task: 1fad071e (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # colour 1 is hard-coded as the "block" colour; bgc and the distractor colour are sampled
    cols = [c for c in range(10) if c != 1]
    bgc, otherc = random.sample(cols, 2)
    return {"bgc": bgc, "otherc": otherc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, otherc) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return min(b, max(a, int(round(a + (b - a) * d))))

    def dneigh(cells):
        out = set()
        for (i, j) in cells:
            out |= {(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}
        return out

    hmax = max(10, min(30, max_h))
    wmax = max(10, min(30, max_w))
    h = unifint(diff_lb, diff_ub, (10, hmax))
    w = unifint(diff_lb, diff_ub, (10, wmax))
    nbl = random.randint(0, 5)
    nobjs = unifint(diff_lb, diff_ub, (nbl, max(nbl, (h * w) // 10)))
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    bcount = 0
    gi = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}
    forb = {1: set(), otherc: set()}
    while succ < nobjs and tr < maxtr:
        tr += 1
        col = random.choice((1, otherc))
        oh = random.randint(1, 3)
        ow = random.randint(1, 3)
        if bcount < nbl:
            col = 1
            oh, ow = 2, 2
        else:
            while col == 1 and oh == ow == 2:
                col = random.choice((1, otherc))
                oh = random.randint(1, 3)
                ow = random.randint(1, 3)
        cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
        if not cands:
            continue
        loci, locj = random.choice(cands)
        bd = {(loci + a, locj + b) for a in range(oh) for b in range(ow)}
        if bd.issubset(inds) and len(dneigh(bd) & forb[col]) == 0:
            succ += 1
            inds -= bd
            forb[col] |= dneigh(bd) | bd
            for (i, j) in bd:
                gi[i][j] = col
            if col == 1 and oh == ow == 2:
                bcount += 1
    go = [[1] * bcount + [bgc] * (5 - bcount)]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    # Rule: a 1x5 tally bar; one cell of colour 1 per isolated 2x2 block of colour 1,
    # the rest of the bar is the background colour.
    # (The golf entry counts all-1 2x2 *windows*, which overcounts 2x3/3x3 blue
    # rectangles; the generator makes each blue rectangle its own isolated component, so the
    # unit to count is the component that is exactly a 2x2 square — the same reading as
    # the maker. The output construction is identical too, so nothing changes.)
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # background: the canvas colour every example input shares; examples' non-1 tally
    # cells name it directly when available, else fall back to majority colour of I.
    bgc = None
    if examples:
        votes = Counter()
        for ex_i, ex_o in examples:
            ex_i = np.asarray(ex_i, dtype=int)
            ex_o = np.asarray(ex_o, dtype=int)
            for v in ex_o.flatten().tolist():
                if v != 1:
                    votes[v] += 1
        if votes:
            cand = votes.most_common(1)[0][0]
            if cand in set(I.flatten().tolist()):
                bgc = cand
    if bgc is None:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # count colour-1 components that are exactly a 2x2 square
    seen = np.zeros((hi, wi), dtype=bool)
    k = 0
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != 1 or seen[r, c]:
                continue
            comp = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                y, x = q.popleft()
                comp.append((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == 1:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            if len(comp) == 4 and max(rs) - min(rs) == 1 and max(cs) - min(cs) == 1:
                k += 1
    k = min(k, 5)

    ops, sels = [], []
    # shrink canvas to the 1x5 tally bar (full rectangle, row 0 cols 0..4)
    ops.append(33); sels.append([0, 0, 0, 4])
    strip = [int(I[0, j]) for j in range(5)]

    # left segment: one colour-1 cell per 2x2 block (full rectangle of k cells)
    if k > 0 and any(strip[j] != 1 for j in range(k)):
        ops.append(1); sels.append([0, 0, 0, k - 1])
    # right segment: the remaining cells are background (full rectangle)
    if k < 5 and any(strip[j] != bgc for j in range(k, 5)):
        ops.append(bgc); sels.append([0, k, 0, 4 - k])

    ops.append(34); sels.append([0, 0, 0, 4])
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
                        f"num_examples+1 ({num_examples + 1}) for task 1fad071e"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 1fad071e"
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
                                f"for task 1fad071e"
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
                    f"Failed to build a complete episode for task 1fad071e "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"1fad071e-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
