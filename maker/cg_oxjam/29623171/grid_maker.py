"""
ARC Task: 29623171 (RE-ARC) — LLM-generated grid_maker
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
    bgc, linc, fgc = random.sample(range(10), 3)
    return {"bgc": bgc, "linc": linc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, linc=None, fgc=None) -> dict:
    if bgc is None or linc is None or fgc is None:
        bgc, linc, fgc = random.sample(range(10), 3)

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round((b - a) * lb))
        hi_ = a + int(round((b - a) * ub))
        lo, hi_ = max(a, min(lo, b)), max(a, min(hi_, b))
        if hi_ < lo:
            hi_ = lo
        return random.randint(lo, hi_)

    nh = unifint(diff_lb, diff_ub, (2, 4))
    nw = unifint(diff_lb, diff_ub, (2, 4))
    hmax = max(2, min(6, (max_h + 1) // nh - 1))
    wmax = max(2, min(6, (max_w + 1) // nw - 1))
    h = unifint(diff_lb, diff_ub, (2, hmax))
    w = unifint(diff_lb, diff_ub, (2, wmax))
    if h * w < 6:
        # need at least 1 mark with nmostc <= hw//2 - 1
        h, w = max(h, 2), max(w, 3)
    fullh = h * nh + (nh - 1)
    fullw = w * nw + (nw - 1)
    gi = [[linc] * fullw for _ in range(fullh)]
    go = [[linc] * fullw for _ in range(fullh)]
    inds = [(i, j) for i in range(h) for j in range(w)]
    llocs = [(a, b) for a in range(0, fullh, h + 1) for b in range(0, fullw, w + 1)]
    srcloc = random.choice(llocs)
    nmostc = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 2 - 1)))
    mostc = set(random.sample(inds, nmostc))
    for (i, j) in inds:
        v = fgc if (i, j) in mostc else bgc
        gi[srcloc[0] + i][srcloc[1] + j] = v
        go[srcloc[0] + i][srcloc[1] + j] = fgc
    for rl in llocs:
        if rl == srcloc:
            continue
        noth = unifint(diff_lb, diff_ub, (0, nmostc))
        marks = set(random.sample(inds, noth))
        for (i, j) in inds:
            gi[rl[0] + i][rl[1] + j] = fgc if (i, j) in marks else bgc
            go[rl[0] + i][rl[1] + j] = fgc if noth == nmostc else bgc
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # Grid lines: full rows / columns of one uniform colour.
    sep_rows = {r for r in range(hi) if len(set(I[r, :].tolist())) == 1}
    sep_cols = {c for c in range(wi) if len(set(I[:, c].tolist())) == 1}

    def runs(n, seps):
        out, cur = [], []
        for i in range(n):
            if i in seps:
                if cur:
                    out.append(cur)
                    cur = []
            else:
                cur.append(i)
        if cur:
            out.append(cur)
        return out

    row_groups = runs(hi, sep_rows)
    col_groups = runs(wi, sep_cols)

    cnt = Counter()
    for rg in row_groups:
        for cg in col_groups:
            for r in rg:
                for c in cg:
                    cnt[int(I[r, c])] += 1
    if len(cnt) < 2 or not row_groups or not col_groups:
        ops.append(34)
        sels.append([0, 0, ho - 1, wo - 1])
        return ops, sels
    ordered = cnt.most_common()
    bgc = ordered[0][0]
    fgc = ordered[-1][0]

    # Each block is the unit: its score is how many mark cells it holds.
    blocks = []
    for rg in row_groups:
        for cg in col_groups:
            cells = [(r, c) for r in rg for c in cg]
            n = sum(1 for (r, c) in cells if int(I[r, c]) == fgc)
            blocks.append((cells, n))
    best = max(n for _, n in blocks)

    # Step 1 (verifier reading): wipe every mark back to background at once.
    all_marks = [(r, c) for rg in row_groups for cg in col_groups
                 for r in rg for c in cg if int(I[r, c]) == fgc]
    if all_marks:
        ops.append(int(bgc))
        sels.append(sel_of(all_marks))

    # Step 2: each block reaching the maximum count is filled solid with the mark colour.
    for cells, n in blocks:
        if n == best:
            ops.append(int(fgc))
            sels.append(sel_of(cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task 29623171"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 29623171"
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
                                f"for task 29623171"
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
                    f"Failed to build a complete episode for task 29623171 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"29623171-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
