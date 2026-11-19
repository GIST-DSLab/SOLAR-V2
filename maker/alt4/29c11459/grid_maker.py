"""
ARC Task: 29c11459 (RE-ARC) — LLM-generated grid_maker
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


# ---------------------------------------------------------------- helpers ---
def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    if b < a:
        b = a
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    lo = max(a, min(b, lo))
    hi = max(lo, min(b, hi))
    return random.randint(lo, hi)


# the only discrete structural variant: which axis the marker pairs live on
VARIANTS = [{"mirrored": False}, {"mirrored": True}]


# ----------------------------------------------------------- sample_colors ---
def sample_colors(num_examples=None) -> dict:
    colopts = [c for c in range(10) if c != 5]          # 5 is reserved (hardcoded)
    bgc = random.choice(colopts)
    rem = [c for c in colopts if c != bgc]
    ncols = random.randint(3, len(rem))
    ccols = random.sample(rem, ncols)

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "ccols": ccols, "instance_plan": plan}


# ----------------------------------------------------------------- generate ---
def generate(diff_lb, diff_ub, max_h, max_w, bgc, ccols, mirrored=None, **kwargs) -> dict:
    if mirrored is None:
        mirrored = random.choice([True, False])

    # pre-mirror grid is (h, w); after dmirror it becomes (w, h)
    if mirrored:
        h_ub, w_ub = min(30, max_w), min(29, max_h)
    else:
        h_ub, w_ub = min(30, max_h), min(29, max_w)
    h_ub = max(5, h_ub)
    w_ub = max(5, w_ub)

    h = _unifint(diff_lb, diff_ub, (5, h_ub))
    w = _unifint(diff_lb, diff_ub, (5, w_ub))
    if w % 2 == 0:                                   # the line axis must be odd
        cands = [x for x in (w - 1, w + 1) if 5 <= x <= w_ub]
        w = random.choice(cands) if cands else 5

    remcols = [c for c in ccols if c != bgc and c != 5]
    if len(remcols) < 2:
        remcols = [c for c in range(10) if c not in (bgc, 5)]
    ncols = _unifint(diff_lb, diff_ub, (2, len(remcols)))
    cc = random.sample(remcols, ncols)

    nlocs = _unifint(diff_lb, diff_ub, (1, h))
    locs = random.sample(range(h), nlocs)
    while set(locs).issubset({0, h - 1}):
        locs = random.sample(range(h), nlocs)

    acols, bcols = [], []
    aforb = bforb = -1
    for _ in range(nlocs):
        ac = random.choice([c for c in cc if c != aforb])
        acols.append(ac)
        aforb = ac
        bc = random.choice([c for c in cc if c != bforb])
        bcols.append(bc)
        bforb = bc

    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    mid = w // 2
    for (a, b), loc in zip(zip(acols, bcols), sorted(locs)):
        gi[loc][0] = a
        gi[loc][w - 1] = b
        for c in range(0, mid):
            go[loc][c] = a
        for c in range(mid + 1, w):
            go[loc][c] = b
        go[loc][mid] = 5

    if mirrored:                                      # dmirror == transpose
        gi = [list(r) for r in zip(*gi)]
        go = [list(r) for r in zip(*go)]

    return {
        "input": tuple(tuple(r) for r in gi),
        "output": tuple(tuple(r) for r in go),
    }


# -------------------------------------------------------- derive_operations ---
def derive_operations(I, O, examples=None):
    """
    Rule: every pair of edge markers facing each other across the grid grows into
    a full line.  The marker line itself is REPLICATED across its half of the
    grid (CopyI the whole edge row/column, Paste it onto each row/column of that
    half), and the single middle row/column of the pair gets the colour-5 seam.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # --- orientation + background -------------------------------------------
    inner_rows = I[1:h - 1, :].flatten().tolist()
    vertical = len(set(inner_rows)) == 1          # markers on row 0 / row h-1
    if vertical:
        bgc = int(inner_rows[0])
    else:
        inner_cols = I[:, 1:w - 1].flatten().tolist()
        bgc = int(Counter(inner_cols).most_common(1)[0][0])

    if vertical:
        mid = h // 2
        lines = [c for c in range(w) if I[0, c] != bgc and I[h - 1, c] != bgc]
        halves = [(0, list(range(1, mid))), (h - 1, list(range(mid + 1, h - 1)))]
        for src_row, dst_rows in halves:
            if not dst_rows:
                continue
            # CopyI of the whole marker row: bbox IS exactly the intended cells
            ops.append(28); sels.append([src_row, 0, 0, w - 1])
            for r in dst_rows:
                ops.append(30); sels.append([r, 0, 0, 0])   # paste origin
            if bgc != 0:
                # Paste is transparent to 0, so markers coloured 0 didn't land:
                # repair exactly those replicated cells, column by column.
                zc = [c for c in lines if I[src_row, c] == 0]
                cells = [(r, c) for c in zc for r in dst_rows]
                if cells:
                    ops.append(0); sels.append(sel_of(cells))
        seam = [(mid, c) for c in lines]
        ops.append(5); sels.append(sel_of(seam))
    else:
        mid = w // 2
        lines = [r for r in range(h) if I[r, 0] != bgc and I[r, w - 1] != bgc]
        halves = [(0, list(range(1, mid))), (w - 1, list(range(mid + 1, w - 1)))]
        for src_col, dst_cols in halves:
            if not dst_cols:
                continue
            # CopyI of the whole marker column: bbox IS exactly the intended cells
            ops.append(28); sels.append([0, src_col, h - 1, 0])
            for c in dst_cols:
                ops.append(30); sels.append([0, c, 0, 0])   # paste origin
            if bgc != 0:
                zr = [r for r in lines if I[r, src_col] == 0]
                cells = [(r, c) for r in zr for c in dst_cols]
                if cells:
                    ops.append(0); sels.append(sel_of(cells))
        seam = [(r, mid) for r in lines]
        ops.append(5); sels.append(sel_of(seam))

    ops.append(34); sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 29c11459"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 29c11459"
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
                                f"for task 29c11459"
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
                    f"Failed to build a complete episode for task 29c11459 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"29c11459-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
