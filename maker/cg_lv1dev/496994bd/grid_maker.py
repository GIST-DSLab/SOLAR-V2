"""
ARC Task: 496994bd (RE-ARC) — LLM-generated grid_maker
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
    # Background is the only colour role the rule relies on; the content
    # colours are just copied, whatever they are.
    bgc = random.choice(list(range(1, 10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    from random import choice, sample

    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        d = random.uniform(lb, ub)
        return int(round(a + (b - a) * d))

    cols = list(range(1, 10))
    if bgc is None:
        bgc = choice(cols)
    h = unifint(diff_lb, diff_ub, (3, min(30, max_h)))
    wmax = max(3, min(14, (max_w - 1) // 2))
    w = unifint(diff_lb, diff_ub, (3, wmax))
    remcols = [c for c in cols if c != bgc]
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = sample(remcols, numcols)

    gix = [[bgc] * w for _ in range(h)]
    nc = unifint(diff_lb, diff_ub, (2, h * w - 1))
    bx = {(i, j) for i in range(h) for j in range(w)}
    top = [ij for ij in bx if ij[0] < h // 2]
    bot = [ij for ij in bx if ij[0] > h // 2]
    obj = {}
    a = choice(top); obj[a] = choice(remcols)
    b = choice(bot); obj[b] = choice(remcols)
    bx -= {a, b}
    for _ in range(nc - 2):
        dns = set()
        for (i, j) in obj:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        dns.add((i + di, j + dj))
        cand = list(bx & dns)
        if not cand:
            break
        ch = choice(cand)
        obj[ch] = choice(remcols)
        bx -= {ch}
    for (i, j), c in obj.items():
        gix[i][j] = c
    # push background to the right of each row (order by "is bgc")
    gix = [sorted(row, key=lambda v: v == bgc) for row in gix]

    flag = choice((True, False))
    canv = [[bgc] * w for _ in range(h)]
    mir = [row[::-1] for row in gix]
    if flag:
        gi = [gix[i] + canv[i] for i in range(h)]
        go = [gix[i] + mir[i] for i in range(h)]
    else:
        gi = [gix[i] + [bgc] + canv[i] for i in range(h)]
        go = [gix[i] + [bgc] + mir[i] for i in range(h)]
    if choice((True, False)):
        gi = [r[::-1] for r in gi]
        go = [r[::-1] for r in go]
    if choice((True, False)):
        gi = gi[::-1]
        go = go[::-1]
    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O, examples=None):
    """Reading taken from the golf entry  a[:5] + a[4::-1]:
    the output is the content half followed by the same LINES in reverse
    order, starting from the line next to the seam.  In this RE-ARC
    variant the lines are columns: content column c reappears at the
    mirrored column wi-1-c.  So the unit is one column (a full-height
    line).  We copy each content column out of the input and paste it at
    its mirror position, working outward from the seam the way
    a[4::-1] does.
    All quantities come from I: bgc is the majority colour (the blank
    half makes sure of that), the content half is the half that is not
    entirely background, and the seam is the centre of the grid.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    w = wi // 2
    left_has = bool(np.any(I[:, :w] != bgc))

    if left_has:
        src_cols = list(range(w - 1, -1, -1))   # nearest the seam first
    else:
        src_cols = list(range(wi - w, wi))      # nearest the seam first

    for c in src_cols:
        col = I[:, c]
        if not np.any(col != bgc):
            continue                             # a blank line adds nothing
        dst = wi - 1 - c
        # Full column rectangle: the whole line is the unit that is copied.
        ops.append(28); sels.append(sel_of([(r, c) for r in range(hi)]))
        # Paste origin = top cell of the mirrored column.
        ops.append(30); sels.append(sel_of([(0, dst)]))

    ops.append(34)
    sels.append([0, 0, ho - 1, wo - 1])          # full-grid rectangle
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
                        f"num_examples+1 ({num_examples + 1}) for task 496994bd"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 496994bd"
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
                                f"for task 496994bd"
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
                    f"Failed to build a complete episode for task 496994bd "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"496994bd-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
