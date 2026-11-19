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


# ---------------------------------------------------------------- colors ----
def sample_colors(num_examples=None) -> dict:
    """bgc / linc / fgc are the three colours the original generator samples.

    The rule (block with most marks wins) is colour-independent, but the solver
    must be able to tell background, separator line and mark apart, so all three
    roles are pinned once per episode.  No structural variants exist in this
    task, so there is no instance plan.
    """
    cols = list(range(10))
    bgc, linc, fgc = random.sample(cols, 3)
    return {"bgc": bgc, "linc": linc, "fgc": fgc}


# -------------------------------------------------------------- generator ----
def generate(diff_lb, diff_ub, max_h, max_w, bgc, linc, fgc) -> dict:
    # block counts, then block size capped so the full grid fits max_h/max_w
    nh = unifint(diff_lb, diff_ub, (2, 4))
    nw = unifint(diff_lb, diff_ub, (2, 4))
    while nh > 2 and (max_h - (nh - 1)) // nh < 2:
        nh -= 1
    while nw > 2 and (max_w - (nw - 1)) // nw < 2:
        nw -= 1
    hub = max(2, min(6, (max_h - (nh - 1)) // nh))
    wub = max(2, min(6, (max_w - (nw - 1)) // nw))
    h = unifint(diff_lb, diff_ub, (2, hub))
    w = unifint(diff_lb, diff_ub, (2, wub))

    fullh = h * nh + (nh - 1)
    fullw = w * nw + (nw - 1)

    gi = [[linc] * fullw for _ in range(fullh)]
    go = [[linc] * fullw for _ in range(fullh)]

    llocs = [(a, b) for a in range(0, fullh, h + 1) for b in range(0, fullw, w + 1)]
    inds = [(r, c) for r in range(h) for c in range(w)]

    srcloc = random.choice(llocs)
    nmostc = unifint(diff_lb, diff_ub, (1, (h * w) // 2 - 1))
    mostc = random.sample(inds, nmostc)

    for (a, b) in llocs:
        if (a, b) == srcloc:
            marks = mostc
        else:
            noth = unifint(diff_lb, diff_ub, (0, nmostc))
            marks = random.sample(inds, noth)
        mset = set(marks)
        winner = len(mset) == nmostc
        for (r, c) in inds:
            gi[a + r][b + c] = fgc if (r, c) in mset else bgc
            go[a + r][b + c] = fgc if winner else bgc

    return {
        "input": tuple(tuple(row) for row in gi),
        "output": tuple(tuple(row) for row in go),
    }


# ------------------------------------------------------------ derivation ----
def derive_operations(I, O, examples=None):
    """Rule: the block holding the most marks is coloured in completely with the
    mark colour; every other block has its marks wiped back to background.

    Everything below is measured from I only.  The route a solver takes through
    that rule is genuinely open — where to start, which way to sweep a block in,
    whether to deal with the winner before or after the losers, whether to rub
    the losers out one block at a time or in a single sweep — and those choices
    are taken from features of I (mark total, block size, winner position), so
    the same input always yields the same route and a different O would change
    nothing.
    """
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    # --- separator lines: full rows / columns of one uniform colour ---
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

    # --- block colours: majority inside the blocks is background, other is mark ---
    cnt = Counter()
    for rg in row_groups:
        for cg in col_groups:
            for r in rg:
                for c in cg:
                    cnt[int(I[r, c])] += 1
    if len(cnt) < 2 or not row_groups or not col_groups:
        return [34], [[0, 0, hi - 1, wi - 1]]
    ordered = cnt.most_common()
    bgc = int(ordered[0][0])
    fgc = int(ordered[-1][0])

    # --- the blocks and their mark counts, read off I ---
    blocks = []
    for bi, rg in enumerate(row_groups):
        for bj, cg in enumerate(col_groups):
            marked = [(r, c) for r in rg for c in cg if int(I[r, c]) == fgc]
            blocks.append({"bi": bi, "bj": bj, "rows": rg, "cols": cg, "marked": marked})

    best = max(len(b["marked"]) for b in blocks)
    winners = [b for b in blocks if len(b["marked"]) == best]
    losers = [b for b in blocks if len(b["marked"]) < best and b["marked"]]
    anchor = winners[0]

    # --- route choices, all read out of I ---
    total = sum(len(b["marked"]) for b in blocks)
    bh, bw = len(row_groups[0]), len(col_groups[0])
    nbr, nbc = len(row_groups), len(col_groups)

    order_mode = (total + bw) % 4          # 0 reading, 1 column-wise, 2 backwards, 3 outward from winner
    pass_mode = (best + anchor["bi"] + anchor["bj"]) % 3   # 0 winner first, 1 as encountered, 2 winner last
    fill_style = (bh * bw + best + anchor["bj"]) % 5       # whole block / row sweep / column sweep
    sweep_losers = pass_mode != 1 and (total + nbr + nbc) % 2 == 1

    def ordered_blocks(bs):
        if order_mode == 0:
            return sorted(bs, key=lambda b: (b["bi"], b["bj"]))
        if order_mode == 1:
            return sorted(bs, key=lambda b: (b["bj"], b["bi"]))
        if order_mode == 2:
            return sorted(bs, key=lambda b: (-b["bi"], -b["bj"]))
        return sorted(
            bs,
            key=lambda b: (abs(b["bi"] - anchor["bi"]) + abs(b["bj"] - anchor["bj"]),
                           b["bi"], b["bj"]),
        )

    def emit_fill(b):
        rows, cols = b["rows"], b["cols"]
        if fill_style == 0:
            strips = [[(r, c) for r in rows for c in cols]]          # colour the block in one go
        elif fill_style in (1, 2):
            strips = [[(r, c) for c in cols] for r in rows]          # row by row
            if fill_style == 2:
                strips = strips[::-1]                                # ... from the bottom up
        else:
            strips = [[(r, c) for r in rows] for c in cols]          # column by column
            if fill_style == 4:
                strips = strips[::-1]                                # ... from the right
        for st in strips:
            if all(int(I[r, c]) == fgc for (r, c) in st):
                continue                                             # already that colour: no visible change
            ops.append(fgc)
            sels.append(sel_of(st))

    def emit_erase(bs):
        bs = ordered_blocks(bs)
        if not bs:
            return
        if sweep_losers:
            cells = []
            for b in bs:
                cells.extend(b["marked"])                            # one sweep over every stray mark
            ops.append(bgc)
            sels.append(sel_of(cells))
        else:
            for b in bs:
                ops.append(bgc)                                      # block by block
                sels.append(sel_of(b["marked"]))

    if pass_mode == 0:
        for b in ordered_blocks(winners):
            emit_fill(b)
        emit_erase(losers)
    elif pass_mode == 2:
        emit_erase(losers)
        for b in ordered_blocks(winners):
            emit_fill(b)
    else:
        for b in ordered_blocks(blocks):
            if len(b["marked"]) == best:
                emit_fill(b)
            elif b["marked"]:
                ops.append(bgc)
                sels.append(sel_of(b["marked"]))

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
