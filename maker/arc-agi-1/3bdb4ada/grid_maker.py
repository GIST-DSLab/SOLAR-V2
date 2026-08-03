"""
ARC Task: 3bdb4ada (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    bgc = random.choice(range(10))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            a, b = b, a
        lo = a + int((b - a) * lb)
        hi = a + int((b - a) * ub)
        lo = max(a, min(b, lo))
        hi = max(a, min(b, hi))
        if hi < lo:
            lo, hi = hi, lo
        return random.randint(lo, hi)

    cols = list(range(10))
    h = unifint(diff_lb, diff_ub, (min(10, max_h), max_h))
    w = unifint(diff_lb, diff_ub, (min(10, max_w), max_w))
    remcols = [c for c in cols if c != bgc]

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]

    num = unifint(diff_lb, diff_ub, (1, 8))
    free = set((i, j) for i in range(h) for j in range(w))
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or len(free) == 0:
            break
        if random.choice((True, False)):
            oh = 3
            ow = unifint(diff_lb, diff_ub, (1, max(1, w // 2 - 1))) * 2 + 1
        else:
            ow = 3
            oh = unifint(diff_lb, diff_ub, (1, max(1, h // 2 - 1))) * 2 + 1
        subs = [ij for ij in free if ij[0] < h - oh and ij[1] < w - ow]
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = random.choice(sorted(subs))
        bd = set((i, j) for i in range(loci, loci + oh) for j in range(locj, locj + ow))
        col = random.choice(remcols)
        if bd.issubset(free):
            remcols.remove(col)
            for (i, j) in bd:
                gi[i][j] = col
                go[i][j] = col
            if oh == 3:
                ln = [(loci + 1, j) for j in range(locj + 1, locj + ow, 2)]
            else:
                ln = [(i, locj + 1) for i in range(loci + 1, loci + oh, 2)]
            for (i, j) in ln:
                go[i][j] = bgc
            succ += 1
            free -= bd
        tr += 1

    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    # background: cells that changed I->O are painted background color
    diff = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != O[r, c]]
    if diff:
        bgc = int(O[diff[0][0], diff[0][1]])
    else:
        bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    ops, sels = [], []

    # each rectangle is a unique non-bgc color block; punch alternating holes
    # along its middle row (height 3) or middle column (width 3)
    cells_by_color = {}
    for r in range(hi):
        for c in range(wi):
            v = int(I[r, c])
            if v != bgc:
                cells_by_color.setdefault(v, []).append((r, c))

    objs = []
    for col, cells in cells_by_color.items():
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        oh, ow = r1 - r0 + 1, c1 - c0 + 1
        if oh * ow != len(cells):
            continue
        if oh == 3:
            line = [(r0 + 1, j) for j in range(c0 + 1, c0 + ow, 2)]
        else:
            line = [(i, c0 + 1) for i in range(r0 + 1, r0 + oh, 2)]
        line = [(r, c) for (r, c) in line if O[r, c] == bgc and I[r, c] != bgc]
        if line:
            objs.append(((r0, c0), line))

    for _, line in sorted(objs):
        ops.append(int(bgc))
        sels.append(sel_of(line))

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
                        f"num_examples+1 ({num_examples + 1}) for task 3bdb4ada"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3bdb4ada"
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
                                f"for task 3bdb4ada"
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
                    f"Failed to build a complete episode for task 3bdb4ada "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3bdb4ada-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
