"""
ARC Task: 2281f1f4 (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


# ---------------------------------------------------------------- 1. colors
ROT_VARIANTS = [{"rot_k": 0}, {"rot_k": 1}, {"rot_k": 2}, {"rot_k": 3}]


def sample_colors(num_examples=None) -> dict:
    colopts = [c for c in range(10) if c != 2]          # 2 is the paint color
    bgc = random.choice(colopts)
    dc = random.choice([c for c in colopts if c != bgc])

    n_ex = num_examples if num_examples else 3
    if n_ex >= len(ROT_VARIANTS):
        examples = [dict(v) for v in ROT_VARIANTS]
        examples += [dict(random.choice(ROT_VARIANTS)) for _ in range(n_ex - len(ROT_VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(ROT_VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "dc": dc, "instance_plan": plan}


# ---------------------------------------------------------------- 2. generate
def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc: int, dc: int, rot_k=None) -> dict:
    if rot_k is None:
        rot_k = choice((0, 1, 2, 3))

    if rot_k in (1, 3):                      # 90/270 rotation swaps h and w
        lim = min(30, max_h, max_w)
        h_bounds = (3, lim)
        w_bounds = (3, lim)
    else:
        h_bounds = (3, min(30, max_h))
        w_bounds = (3, min(30, max_w))

    h = unifint(diff_lb, diff_ub, h_bounds)
    w = unifint(diff_lb, diff_ub, w_bounds)

    card_h_bounds = (1, h // 2 + 1)
    card_w_bounds = (1, w // 2 + 1)
    numtop = unifint(diff_lb, diff_ub, card_w_bounds)
    numright = unifint(diff_lb, diff_ub, card_h_bounds)
    if numtop == numright == 1:
        numtop, numright = sample([1, 2], 2)

    tp = sample(interval(0, w - 1, 1), numtop)
    rp = sample(interval(1, h, 1), numright)
    res = combine(apply(lbind(astuple, 0), tp), apply(rbind(astuple, w - 1), rp))

    gi = fill(canvas(bgc, (h, w)), dc, res)
    go = fill(gi, 2, product(rp, tp))

    rotf = (identity, rot90, rot180, rot270)[rot_k]
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


# ---------------------------------------------------------------- 3. ops
def derive_operations(I, O):
    """
    Rule (read off I alone):
      * one non-background 'marker' color sits on two perpendicular border
        lines: a border ROW and a border COLUMN.
      * their meeting grid-corner is the corner that shares its row or its
        column with the MOST markers (and is itself background).
      * every marker on the border column projects the whole border-row marker
        pattern into its own row -> paint those intersections with 2.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]                 # canvas color the generator fills
    fg = [c for c in cnt if c != bgc]
    if not fg:
        ops.append(34); sels.append([0, 0, h - 1, w - 1])
        return ops, sels
    mc = min(fg, key=lambda c: cnt[c])             # marker color (leastcolor)

    marks = [(r, c) for r in range(h) for c in range(w) if I[r, c] == mc]

    # --- locate the meeting corner of the two marker lines -----------------
    best, corner = -1, (0, 0)
    for cr in (0, h - 1):
        for cc in (0, w - 1):
            if I[cr, cc] != bgc:
                continue
            k = sum(1 for (r, c) in marks if r == cr or c == cc)
            if k > best:
                best, corner = k, (cr, cc)
    cr, cc = corner

    # markers lying on the corner's column line -> the projected rows
    proj_rows = sorted({r for (r, c) in marks if c == cc and r != cr})
    # markers lying on the corner's row line -> the pattern of columns
    pat_cols = sorted({c for (r, c) in marks if r == cr and c != cc})

    # --- one op per projected line: the whole pattern stamped in that row ---
    for r in proj_rows:
        cells = [(r, c) for c in pat_cols if I[r, c] != 2]
        if cells:
            ops.append(2)
            sels.append(sel_of(cells))

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
                        f"num_examples+1 ({num_examples + 1}) for task 2281f1f4"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 2281f1f4"
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
                                f"for task 2281f1f4"
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
                    f"Failed to build a complete episode for task 2281f1f4 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"2281f1f4-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
