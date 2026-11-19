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
    # Only the background has to stay the same across the episode. Mirroring
    # works the same way whatever the foreground colours are.
    bgc = random.choice(list(range(1, 10)))
    return {"bgc": bgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round(lb * (b - a)))
        hi = a + int(round(ub * (b - a)))
        return random.randint(min(lo, hi), max(lo, hi))

    if bgc is None:
        bgc = random.choice(list(range(1, 10)))
    cols = list(range(1, 10))
    h = unifint(diff_lb, diff_ub, (3, min(30, max_h)))
    w_cap = max(3, min(14, (max_w - 1) // 2))
    w = unifint(diff_lb, diff_ub, (3, w_cap))
    remcols = [c for c in cols if c != bgc]
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = random.sample(remcols, numcols)
    nc = unifint(diff_lb, diff_ub, (2, h * w - 1))

    bx = {(i, j) for i in range(h) for j in range(w)}
    top = [ij for ij in bx if ij[0] < h // 2]
    bot = [ij for ij in bx if ij[0] > h // 2]
    obj = {}
    p1 = random.choice(top)
    obj[p1] = random.choice(remcols)
    p2 = random.choice(bot)
    obj[p2] = random.choice(remcols)
    for _ in range(nc - 2):
        dns = set()
        for (i, j) in obj:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        dns.add((i + di, j + dj))
        cand = sorted(bx & dns)
        if not cand:
            break
        ch = random.choice(cand)
        obj[ch] = random.choice(remcols)
        bx = bx - {ch}

    gix = [[bgc] * w for _ in range(h)]
    for (i, j), c in obj.items():
        gix[i][j] = c
    # Within each row, non-background cells first, background cells last
    # (stable sort).
    gix = [sorted(row, key=lambda v: v == bgc) for row in gix]

    flag = random.choice((True, False))
    gi, go = [], []
    for row in gix:
        mir = row[::-1]
        if flag:
            gi.append(row + [bgc] * w)
            go.append(row + mir)
        else:
            gi.append(row + [bgc] + [bgc] * w)
            go.append(row + [bgc] + mir)
    if random.choice((True, False)):
        gi = [r[::-1] for r in gi]
        go = [r[::-1] for r in go]
    if random.choice((True, False)):
        gi = gi[::-1]
        go = go[::-1]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """Mirror the content block into the blank block one COLUMN at a time.

    The grid has two half-blocks of width w = wi // 2. When wi is odd there is
    one background column between them. One block holds the content and the
    other is blank. The mirror sends column c to column wi-1-c. So each content
    column is copied from the INPUT and pasted at its mirror column. Every
    column that holds any non-background cell gets its own copy and paste.
    Columns that are all background are skipped, because the blank target
    already looks like that. Colours are 1..9, so no cell is 0 and the
    transparent Paste writes the whole column.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # Background is the strict majority colour: one full half-block is blank.
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    w = wi // 2
    left_has = bool(np.any(I[:, 0:w] != bgc))
    src_cols = list(range(0, w)) if left_has else list(range(wi - w, wi))
    # Start at the grid edge and work inward toward the middle.
    if not left_has:
        src_cols = src_cols[::-1]

    for c in src_cols:
        if not np.any(I[:, c] != bgc):
            continue  # an all-background column changes nothing at its mirror
        dst_c = wi - 1 - c
        # Full-column rectangle: the whole column, background included, is
        # the unit being copied.
        col_cells = [(r, c) for r in range(hi)]
        ops.append(28); sels.append(sel_of(col_cells))       # CopyI this column
        ops.append(30); sels.append(sel_of([(0, dst_c)]))    # Paste at mirror column

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
