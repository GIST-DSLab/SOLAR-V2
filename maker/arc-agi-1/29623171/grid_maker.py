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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    """bgc = block background, linc = separator-line color, fgc = the sparse mark color."""
    cols = list(range(10))
    bgc, linc, fgc = random.sample(cols, 3)
    return {"bgc": bgc, "linc": linc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, fgc) -> dict:
    def uf(lb, ub, bounds):
        a, b = bounds
        if b <= a:
            return a
        ha = a - 1 + max(1, int(lb * (b - a + 1)))
        hb = a - 1 + max(1, int(ub * (b - a + 1)))
        ha = max(a, min(b, ha))
        hb = max(a, min(b, hb))
        if hb < ha:
            ha, hb = hb, ha
        return random.randint(ha, hb)

    # number of blocks per axis (2..4), constrained so that h,w >= 2 still fit
    nh_max = min(4, max(2, (max_h + 1) // 3))
    nw_max = min(4, max(2, (max_w + 1) // 3))
    nh = uf(diff_lb, diff_ub, (2, nh_max))
    nw = uf(diff_lb, diff_ub, (2, nw_max))

    h_max = min(6, max(2, (max_h - (nh - 1)) // nh))
    w_max = min(6, max(2, (max_w - (nw - 1)) // nw))
    h = uf(diff_lb, diff_ub, (2, h_max))
    w = uf(diff_lb, diff_ub, (2, w_max))

    fullh = h * nh + (nh - 1)
    fullw = w * nw + (nw - 1)

    inds = [(r, c) for r in range(h) for c in range(w)]
    nmostc = uf(diff_lb, diff_ub, (1, max(1, (h * w) // 2 - 1)))

    # canvas of separator color; blocks carved out of it
    gi = [[linc] * fullw for _ in range(fullh)]
    go = [[linc] * fullw for _ in range(fullh)]

    llocs = [(a, b) for a in range(0, fullh, h + 1) for b in range(0, fullw, w + 1)]
    srcloc = random.choice(llocs)

    counts = {}
    marks = {}
    for loc in llocs:
        if loc == srcloc:
            n = nmostc
        else:
            n = uf(diff_lb, diff_ub, (0, nmostc))
        counts[loc] = n
        marks[loc] = set(random.sample(inds, n)) if n > 0 else set()

    best = max(counts.values())

    for (a, b) in llocs:
        won = counts[(a, b)] == best
        mk = marks[(a, b)]
        for (r, c) in inds:
            gi[a + r][b + c] = fgc if (r, c) in mk else bgc
            go[a + r][b + c] = fgc if won else bgc

    return {"input": gi, "output": go}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # --- separator lines: full rows / columns of one uniform colour ---
    sep_rows = [r for r in range(hi) if len(set(I[r, :].tolist())) == 1]
    sep_cols = [c for c in range(wi) if len(set(I[:, c].tolist())) == 1]
    sep_rows_s, sep_cols_s = set(sep_rows), set(sep_cols)

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

    row_groups = runs(hi, sep_rows_s)
    col_groups = runs(wi, sep_cols_s)

    # --- block colours: majority inside blocks is background, the other is the mark ---
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

    # --- per-block mark counts, measured from I ---
    blocks = []
    for rg in row_groups:
        for cg in col_groups:
            cells = [(r, c) for r in rg for c in cg]
            marked = [(r, c) for (r, c) in cells if int(I[r, c]) == fgc]
            blocks.append((cells, marked))
    best = max(len(m) for _, m in blocks)

    # --- losers: erase their marks back to background, block by block ---
    for cells, marked in blocks:
        if len(marked) < best and marked:
            ops.append(int(bgc))
            sels.append(sel_of(marked))

    # --- winners: flood the whole block with the mark colour ---
    for cells, marked in blocks:
        if len(marked) == best:
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
