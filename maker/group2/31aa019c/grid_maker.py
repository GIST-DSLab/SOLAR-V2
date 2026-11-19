"""
ARC Task: 31aa019c (RE-ARC) — LLM-generated grid_maker
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
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # Only the background is a fixed episode-level role; the target colour is
    # whichever colour appears exactly once, and the halo is always colour 2.
    bgc = random.choice(list(range(10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    cols = list(range(10))
    if bgc is None:
        bgc = random.choice(cols)

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + (b - a) * lb
        hi = a + (b - a) * ub
        return int(round(random.uniform(lo, hi)))

    while True:
        h = unifint(diff_lb, diff_ub, (5, max_h))
        w = unifint(diff_lb, diff_ub, (5, max_w))
        h = max(5, min(h, max_h))
        w = max(5, min(w, max_w))
        inds = [(i, j) for i in range(h) for j in range(w)]
        mp = (h * w) // 2 - 1
        ncols = unifint(diff_lb, diff_ub, (2, min(9, mp // 2 - 1)))
        ncols = max(2, min(ncols, 9))
        chcols = random.sample(cols, ncols)
        trgcol = chcols[0]
        chcols = chcols[1:]
        dic = {c: set() for c in chcols}
        nnoise = unifint(diff_lb, diff_ub, (2 * (ncols - 1), mp))
        locc = random.choice(inds)
        inds.remove(locc)
        if nnoise > len(inds):
            continue
        noise = random.sample(inds, nnoise)
        ok = True
        for _ in range(2):
            for c in chcols:
                if not inds:
                    ok = False
                    break
                ij = random.choice(inds)
                dic[c].add(ij)
                inds.remove(ij)
        if not ok:
            continue
        for ij in noise:
            c = random.choice(chcols)
            dic[c].add(ij)
            if ij in inds:
                inds.remove(ij)
        gi = [[bgc] * w for _ in range(h)]
        gi[locc[0]][locc[1]] = trgcol
        for c, ss in dic.items():
            for (i, j) in ss:
                gi[i][j] = c
        gi[locc[0]][locc[1]] = trgcol
        cnt = Counter(v for row in gi for v in row)
        singles = [c for c in cnt if cnt[c] == 1]
        if len(singles) == 1:
            break

    cnt = Counter(v for row in gi for v in row)
    lc = min(cnt.keys(), key=lambda k: cnt[k])
    mc = max(cnt.keys(), key=lambda k: cnt[k])
    pos = [(i, j) for i in range(h) for j in range(w) if gi[i][j] == lc][0]
    go = [[mc] * w for _ in range(h)]
    go[pos[0]][pos[1]] = lc
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            a, b = pos[0] + di, pos[1] + dj
            if 0 <= a < h and 0 <= b < w:
                go[a][b] = 2
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    cnt = Counter(I.flatten().tolist())
    bgc = max(cnt.keys(), key=lambda k: (cnt[k], -k))
    singles = [c for c in cnt if cnt[c] == 1]
    trg = singles[0] if len(singles) == 1 else min(cnt.keys(), key=lambda k: (cnt[k], k))
    r, c = [int(v) for v in np.argwhere(I == trg)[0]]

    ops, sels = [], []

    # 1. Clear the noise one colour at a time: every cell of each noise colour
    #    is repainted to the background (unit = "all cells of one colour").
    noise_cols = sorted([k for k in cnt if k != bgc and k != trg],
                        key=lambda k: (-cnt[k], k))
    for k in noise_cols:
        cells = [(int(a), int(b)) for a, b in np.argwhere(I == k)]
        ops.append(int(bgc))
        sels.append(sel_of(cells))

    # 2. Draw the colour-2 halo: the 8 neighbours of the unique cell (clipped).
    ring = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            a, b = r + dr, c + dc
            if 0 <= a < hi and 0 <= b < wi:
                ring.append((a, b))
    if bgc != 2 and ring:
        ops.append(2)
        sels.append(sel_of(ring))

    ops.append(34)
    sels.append([0, 0, hi - 1, wi - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 31aa019c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 31aa019c"
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
                                f"for task 31aa019c"
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
                    f"Failed to build a complete episode for task 31aa019c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"31aa019c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
