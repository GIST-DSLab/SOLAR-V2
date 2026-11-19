"""
ARC Task: 2204b7a8 (RE-ARC) — LLM-generated grid_maker
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


VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    remcols = [c for c in cols if c != bgc]
    ccol = random.choice(remcols)
    remcols2 = [c for c in remcols if c != ccol]
    c1 = random.choice(remcols2)
    c2 = random.choice([c for c in remcols2 if c != c1])
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "ccol": ccol, "c1": c1, "c2": c2, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, ccol, c1, c2, transpose=None) -> dict:
    if transpose is None:
        transpose = random.choice([True, False])

    def unif(lo, hi):
        lo, hi = int(lo), int(hi)
        if hi <= lo:
            return lo
        d = random.uniform(diff_lb, diff_ub)
        return max(lo, min(hi, int(round(lo + d * (hi - lo)))))

    # when transposed, final grid is (w, h): swap bounds
    hb = max_w if transpose else max_h
    wb = max_h if transpose else max_w
    while True:
        h = unif(4, min(30, hb))
        w = unif(4, min(30, wb))
        inds = [(i, j) for i in range(h) for j in range(1, w - 1)]
        nc_hi = (h * (w - 2)) // 2 - 1
        if nc_hi < 1:
            continue
        nc = unif(1, nc_hi)
        locs = random.sample(inds, nc)
        if w % 2 == 1:
            locs = [(i, j) for (i, j) in locs if j != w // 2]
        if not locs:
            continue
        gi = np.full((h, w), bgc, dtype=int)
        gi[:, 0] = c1
        gi[:, w - 1] = c2
        for i, j in locs:
            gi[i, j] = ccol
        go = gi.copy()
        for i, j in locs:
            go[i, j] = c1 if j < w // 2 else c2
        if len(set(gi.flatten().tolist())) == 4:
            break
    if transpose:
        gi = gi.T.copy()
        go = go.T.copy()
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O):
    # The code-golf reading (each half of the grid is paired with the border
    # line on its side; that border's colour goes onto the half's marker cells)
    # matches the maker's reading: the unit is a half, keyed to its nearest
    # border. So the operations are the same: one recolour per half.
    I = np.asarray(I, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # Border lines are either the first/last column or the first/last row.
    # A border column is a single colour. In the row case, column 0 holds c1,
    # then interior cells, then c2, so it is never a single colour.
    vertical = (len(set(I[:, 0].tolist())) == 1) and (len(set(I[:, -1].tolist())) == 1)

    if vertical:
        col_a, col_b = int(I[0, 0]), int(I[0, -1])
        interior = I[:, 1:w - 1]
        half = w // 2
        cells = [(r, c) for r in range(h) for c in range(1, w - 1)]
        side_a = lambda r, c: c < half
    else:
        col_a, col_b = int(I[0, 0]), int(I[-1, 0])
        interior = I[1:h - 1, :]
        half = h // 2
        cells = [(r, c) for r in range(1, h - 1) for c in range(w)]
        side_a = lambda r, c: r < half

    cnt = Counter(interior.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    others = [c for c in cnt if c != bgc]
    marker = others[0] if others else bgc

    group_a, group_b = [], []
    for r, c in cells:
        if int(I[r, c]) == marker and marker != bgc:
            (group_a if side_a(r, c) else group_b).append((r, c))

    # marker cells in the first half take that half's border colour
    if group_a:
        ops.append(col_a)
        sels.append(sel_of(sorted(group_a)))
    # marker cells in the second half take the other border's colour
    if group_b:
        ops.append(col_b)
        sels.append(sel_of(sorted(group_b)))

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
                        f"num_examples+1 ({num_examples + 1}) for task 2204b7a8"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 2204b7a8"
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
                                f"for task 2204b7a8"
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
                    f"Failed to build a complete episode for task 2204b7a8 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"2204b7a8-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
