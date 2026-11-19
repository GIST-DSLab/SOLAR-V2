"""
ARC Task: ff28f65a (RE-ARC) — LLM-generated grid_maker
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
    # Colours 1 (pips) and 2 (counted objects) are fixed by the task; only the background is sampled.
    cols = [c for c in range(10) if c not in (1, 2)]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        lo = a + int(round(lb * (b - a)))
        hi = a + int(round(ub * (b - a)))
        lo, hi = min(lo, hi), max(lo, hi)
        return random.randint(max(a, lo), min(b, max(lo, hi)))

    cols = [c for c in range(10) if c not in (1, 2)]
    if bgc is None:
        bgc = random.choice(cols)
    mpr = {1: (0, 0), 2: (0, 2), 3: (1, 1), 4: (2, 0), 5: (2, 2)}
    h = unifint(diff_lb, diff_ub, (3, max(3, max_h)))
    w = unifint(diff_lb, diff_ub, (3, max(3, max_w)))
    nred = random.randint(1, 5)
    gi = [[bgc] * w for _ in range(h)]
    inds = {(i, j) for i in range(h) for j in range(w)}

    def place(n, colfn):
        succ, tr, maxtr = 0, 0, 5 * n
        nonlocal inds
        while tr < maxtr and succ < n:
            tr += 1
            oh = random.randint(1, h // 2 + 1)
            ow = random.randint(1, w // 2 + 1)
            cands = [ij for ij in inds if ij[0] <= h - oh and ij[1] <= w - ow]
            if not cands:
                continue
            loci, locj = random.choice(cands)
            bd = {(i, j) for i in range(loci, loci + oh) for j in range(locj, locj + ow)}
            if bd.issubset(inds):
                succ += 1
                nb = set()
                for (i, j) in bd:
                    for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nb.add((i + di, j + dj))
                inds = (inds - bd) - nb
                c = colfn()
                for (i, j) in bd:
                    gi[i][j] = c
        return succ

    nblue = place(nred, lambda: 2)
    namt = unifint(diff_lb, diff_ub, (0, nred * 2))
    remcols = [c for c in cols if c != bgc]
    if namt > 0:
        place(namt, lambda: random.choice(remcols))
    go = [[bgc] * 3 for _ in range(3)]
    for k in range(nblue):
        r, c = mpr[k + 1]
        go[r][c] = 1
    return {"input": gi, "output": go}


def derive_operations(I, O=None, examples=None):
    # Reading: count the red objects, light that many pips at every other slot
    # of a 1x9 ribbon, then fold the ribbon into a 3x3 board (row by row).
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    cnt = Counter(int(x) for x in I.flatten() if int(x) != 2)
    bgc = cnt.most_common(1)[0][0] if cnt else 0

    # count 4-connected red components
    seen = np.zeros((hi, wi), dtype=bool)
    nred = 0
    for r in range(hi):
        for c in range(wi):
            if I[r, c] == 2 and not seen[r, c]:
                nred += 1
                q = deque([(r, c)]); seen[r, c] = True
                while q:
                    y, x = q.popleft()
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < hi and 0 <= nx < wi and not seen[ny, nx] and I[ny, nx] == 2:
                            seen[ny, nx] = True; q.append((ny, nx))
    nred = min(nred, 5)

    ops, sels = [], []

    # 1. cut the canvas down to a 1x9 ribbon (top row of I, zero-padded past the width)
    ops.append(33); sels.append([0, 0, 0, 8])   # full 1x9 rectangle
    strip = [int(I[0, c]) if c < wi else 0 for c in range(9)]

    # 2. blank the ribbon to background (whole row) if it is not already background
    if any(v != bgc for v in strip):
        ops.append(bgc); sels.append([0, 0, 0, 8])
        strip = [bgc] * 9

    # 3. light one pip per red object at ribbon slots 0, 2, 4, ...
    for k in range(nred):
        ops.append(1); sels.append(sel_of([(0, 2 * k)]))
        strip[2 * k] = 1

    # 4. open room for the fold: extend the canvas to 3x9 (new rows are zero padding)
    ops.append(33); sels.append([0, 0, 2, 8])   # full 3x9 rectangle

    # 5. fold: ribbon segment j (cols 3j..3j+2) slides down j rows and left 3j cols
    for j in (1, 2):
        seg = [(0, c) for c in range(3 * j, 3 * j + 3)]
        if all(strip[c] == 0 for _, c in seg):
            continue  # nothing to carry (bgc 0 and no pips there): a move would be invisible
        first = True
        for _ in range(j):
            ops.append(21); sels.append(sel_of(seg) if first else sel_of([]))
            first = False
        for _ in range(3 * j):
            ops.append(23); sels.append(sel_of([]))

    # 6. crop to the folded 3x3 board
    ops.append(33); sels.append([0, 0, 2, 2])   # full 3x3 rectangle
    ops.append(34); sels.append([0, 0, 2, 2])
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
                        f"num_examples+1 ({num_examples + 1}) for task ff28f65a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task ff28f65a"
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
                                f"for task ff28f65a"
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
                    f"Failed to build a complete episode for task ff28f65a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"ff28f65a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
