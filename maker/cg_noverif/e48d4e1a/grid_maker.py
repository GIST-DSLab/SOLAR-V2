"""
ARC Task: e48d4e1a (RE-ARC) — LLM-generated grid_maker
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
    # fgc must be non-zero: ARCLE's object buffer drops 0 cells, so a 0-coloured
    # line could not be Moved.
    bgc = random.choice(range(10))
    fgc = random.choice([c for c in range(1, 10) if c != bgc])
    dotc = random.choice([c for c in range(10) if c not in (bgc, fgc)])
    return {"bgc": bgc, "fgc": fgc, "dotc": dotc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None, dotc=None) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        d = random.uniform(lb, ub)
        return min(b, max(a, a + round(d * (b - a))))
    if bgc is None or fgc is None or dotc is None:
        bgc = random.choice(range(10))
        fgc = random.choice([c for c in range(1, 10) if c != bgc])
        dotc = random.choice([c for c in range(10) if c not in (bgc, fgc)])
    h = unifint(diff_lb, diff_ub, (3, max_h))
    w = unifint(diff_lb, diff_ub, (3, max_w))
    loci = random.randint(1, h - 2)
    locj = random.randint(1, w - 2)
    inds = [(i, j) for i in range(loci) for j in range(locj)]
    maxn = min(min(h - loci - 1, w - locj - 1), len(inds))
    nn = unifint(diff_lb, diff_ub, (1, maxn))
    ss = random.sample(inds, nn)
    gi = np.full((h, w), bgc, dtype=int)
    go = np.full((h, w), bgc, dtype=int)
    gi[loci, :] = fgc
    gi[:, locj] = fgc
    for (i, j) in ss:
        gi[i, j] = dotc
    go[loci + nn, :] = fgc
    go[:, locj + nn] = fgc
    k = random.choice((0, 1, 2, 3))   # identity / rot270 / rot180 / rot90
    gi = np.rot90(gi, k)
    go = np.rot90(go, k)
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    """Reading (shared with the golf entry, which rolls the whole grid by n on
    both axes): the dots are a COUNTER -- their number n is the displacement --
    and the cross is TRANSLATED by n diagonally, away from the dots' side.
    Unlike the maker (erase old cross, paint new one), the cross is treated as
    two lines that each keep their full length: the full row slides n cells
    vertically and the full column slides n cells horizontally, both as Move
    chains.  The dots are consumed (erased)."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape

    def sign(x):
        return 0 if x == 0 else (1 if x > 0 else -1)

    # line colour: the colour forming both a full row and a full column of I
    hrow = vcol = fgc = None
    for r in range(hi):
        v = int(I[r, 0])
        if np.all(I[r, :] == v):
            cs = [c for c in range(wi) if np.all(I[:, c] == v)]
            if cs:
                hrow, vcol, fgc = r, cs[0], v
                break
    # background: most frequent colour off the cross (dots are always fewer)
    off = Counter(int(I[r, c]) for r in range(hi) for c in range(wi)
                  if r != hrow and c != vcol)
    bgc = off.most_common(1)[0][0]
    dots = [(r, c) for r in range(hi) for c in range(wi)
            if r != hrow and c != vcol and I[r, c] != bgc]
    nn = len(dots)
    di = sign(hrow - sum(r for r, _ in dots) / nn)
    dj = sign(vcol - sum(c for _, c in dots) / nn)
    new_hrow = hrow + di * nn
    new_vcol = vcol + dj * nn

    ops, sels = [], []
    # 1. consume the dots (erase them to background)
    ops.append(bgc); sels.append(sel_of(dots))

    # 2. slide the horizontal line (whole row) vertically by nn, away from the dots
    row_cells = [(hrow, c) for c in range(wi)]
    mop = 21 if di > 0 else 20
    ops.append(mop); sels.append(sel_of(row_cells))         # grab the row
    for _ in range(nn - 1):
        ops.append(mop); sels.append(sel_of([]))            # keep sliding it
    if bgc != 0:                                            # its old footprint reads 0
        ops.append(bgc); sels.append(sel_of(row_cells))

    # 3. slide the vertical line horizontally by nn, away from the dots.
    #    The line = line-coloured cells of column vcol as the grid stands now
    #    (its crossing with the old row was vacated in step 2).
    col_cells = [(r, vcol) for r in range(hi) if r != hrow]
    mop = 22 if dj > 0 else 23
    ops.append(mop); sels.append(sel_of(col_cells))         # grab the column
    for _ in range(nn - 1):
        ops.append(mop); sels.append(sel_of([]))
    if bgc != 0:
        ops.append(bgc); sels.append(sel_of(col_cells))

    # 4. re-join the cross: where each moved line passes the other line's old
    #    position, the crossing cell was vacated -- restore it with the line colour.
    ops.append(fgc); sels.append(sel_of([(hrow, new_vcol), (new_hrow, vcol)]))

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task e48d4e1a"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task e48d4e1a"
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
                                f"for task e48d4e1a"
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
                    f"Failed to build a complete episode for task e48d4e1a "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"e48d4e1a-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
