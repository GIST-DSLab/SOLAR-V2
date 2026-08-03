"""
ARC Task: 5c2c9af4 (RE-ARC) — LLM-generated grid_maker
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

MODES = ["box", "hline", "vline"]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(MODES):
        examples = [{"mode": m} for m in MODES]
        examples += [{"mode": random.choice(MODES)} for _ in range(n_ex - len(MODES))]
        random.shuffle(examples)
    else:
        examples = [{"mode": m} for m in random.sample(MODES, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "fgc": fgc, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, mode=None) -> dict:
    if mode is None:
        mode = random.choice(MODES)

    def odd_in(lo, hi):
        if hi < lo:
            hi = lo
        n = (hi - lo) // 2
        k = unifint(diff_lb, diff_ub, (0, n))
        return lo + 2 * k

    h = unifint(diff_lb, diff_ub, (5, max_h))
    w = unifint(diff_lb, diff_ub, (5, max_w))
    hodd = h if h % 2 == 1 else h - 1
    wodd = w if w % 2 == 1 else w - 1

    if mode == "hline":
        boxh, boxw = 1, odd_in(3, wodd)
    elif mode == "vline":
        boxh, boxw = odd_in(3, hodd), 1
    else:
        boxh, boxw = odd_in(3, hodd), odd_in(3, wodd)

    loci = random.randint(0, h - boxh)
    locj = random.randint(0, w - boxw)

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    A = (loci, locj)
    B = (loci + boxh - 1, locj + boxw - 1)
    cpi, cpj = loci + boxh // 2, locj + boxw // 2
    for (r, c) in {A, B, (cpi, cpj)}:
        gi[r][c] = fgc
        go[r][c] = fgc

    f1, f2 = boxh // 2, boxw // 2
    if f1 == 0:
        for c in range(w):
            go[cpi][c] = fgc
    elif f2 == 0:
        for r in range(h):
            go[r][cpj] = fgc
    else:
        k = 1
        while True:
            r0, r1 = cpi - k * f1, cpi + k * f1
            c0, c1 = cpj - k * f2, cpj + k * f2
            if r0 < 0 and r1 >= h and c0 < 0 and c1 >= w:
                break
            for c in range(max(c0, 0), min(c1, w - 1) + 1):
                if 0 <= r0 < h:
                    go[r0][c] = fgc
                if 0 <= r1 < h:
                    go[r1][c] = fgc
            for r in range(max(r0, 0), min(r1, h - 1) + 1):
                if 0 <= c0 < w:
                    go[r][c0] = fgc
                if 0 <= c1 < w:
                    go[r][c1] = fgc
            k += 1
            if k > 2 * (h + w) + 5:
                break

    if mode == "box" and random.choice((True, False)):
        gi = [list(x) for x in zip(*gi)]
        go = [list(x) for x in zip(*go)]

    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # the three marker cells are the rare color; background fills the rest
    cnt = Counter(I.flatten().tolist())
    fgc = min(cnt, key=lambda c: cnt[c])
    pts = sorted(tuple(p) for p in np.argwhere(I == fgc).tolist())

    if len(pts) >= 3:
        A, B = pts[0], pts[-1]
        cpr, cpc = (A[0] + B[0]) // 2, (A[1] + B[1]) // 2
        f1, f2 = (B[0] - A[0]) // 2, (B[1] - A[1]) // 2

        targets = []
        if f1 == 0:                      # markers collinear in a row -> whole row
            targets.append([(cpr, c) for c in range(wi)])
        elif f2 == 0:                    # collinear in a column -> whole column
            targets.append([(r, cpc) for r in range(hi)])
        else:                            # concentric box rings, innermost outward
            k = 1
            while True:
                r0, r1 = cpr - k * f1, cpr + k * f1
                c0, c1 = cpc - k * f2, cpc + k * f2
                if r0 < 0 and r1 >= hi and c0 < 0 and c1 >= wi:
                    break
                cells = set()
                for c in range(max(c0, 0), min(c1, wi - 1) + 1):
                    if 0 <= r0 < hi:
                        cells.add((r0, c))
                    if 0 <= r1 < hi:
                        cells.add((r1, c))
                for r in range(max(r0, 0), min(r1, hi - 1) + 1):
                    if 0 <= c0 < wi:
                        cells.add((r, c0))
                    if 0 <= c1 < wi:
                        cells.add((r, c1))
                targets.append(sorted(cells))
                k += 1
                if k > 2 * (hi + wi) + 5:
                    break

        for cells in targets:
            paint = [(r, c) for (r, c) in cells if I[r, c] != O[r, c]]
            if paint:
                ops.append(int(fgc))
                sels.append(sel_of(paint))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 5c2c9af4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 5c2c9af4"
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
                                f"for task 5c2c9af4"
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
                    f"Failed to build a complete episode for task 5c2c9af4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"5c2c9af4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
