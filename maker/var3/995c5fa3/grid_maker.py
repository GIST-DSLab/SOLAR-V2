"""
ARC Task: 995c5fa3 (RE-ARC) — LLM-generated grid_maker
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
# ── shared knowledge: the four 4x4 stamps and the badge each one earns ────────
# o1 solid square           -> 2   (the block is a single colour)
# o2 hollow box             -> 8   (none of the other three tests fire)
# o3 square with side dents -> 3   (top-left cell differs from the one below it)
# o4 square with a 2x2 bite -> 4   (the only stamp that is not up-down symmetric)
import random
import numpy as np

try:
    from maker.sel_helpers import sel_of
except Exception:                                    # stand-alone fallback
    def sel_of(cells):
        uniq = sorted({(int(r), int(c)) for r, c in cells})
        return {"cells": [[r, c] for r, c in uniq]}


def _shapes():
    full = {(r, c) for r in range(4) for c in range(4)}
    o1 = frozenset(full)                                                       # -> 2
    o2 = frozenset({(r, c) for (r, c) in full if r in (0, 3) or c in (0, 3)})   # -> 8
    o3 = frozenset(full - {(1, 0), (2, 0), (1, 3), (2, 3)})                     # -> 3
    o4 = frozenset(full - {(2, 1), (2, 2), (3, 1), (3, 2)})                     # -> 4
    return [(o1, 2), (o2, 8), (o3, 3), (o4, 4)]


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = int(a + (1 - diff_ub) * (b - a))
    hi = int(a + (1 - diff_lb) * (b - a))
    lo = max(a, min(lo, b))
    hi = max(a, min(hi, b))
    if lo > hi:
        lo, hi = hi, lo
    return random.randint(lo, hi)


# ---------------------------------------------------------------- 1. colours
def sample_colors(num_examples=None) -> dict:
    # Only the background is a colour role: the badge is read off the stamp's
    # SHAPE, never off the colour it happens to be inked in, so per-block
    # colours may stay free.  What must be planned is which stamps are shown,
    # so every shape -> badge pair the test can use has been demonstrated.
    bgc = random.choice(list(range(10)))
    n_ex = num_examples if num_examples else 3
    variants = [{"shape_offset": i} for i in range(4)]
    if n_ex >= len(variants):
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(variants, n_ex)]
    plan = examples + [dict(random.choice(examples))]     # test reuses a shown variant
    return {"bgc": bgc, "instance_plan": plan}


# ---------------------------------------------------------------- 2. generate
def generate(diff_lb, diff_ub, max_h, max_w, bgc, shape_offset=None) -> dict:
    mpr = _shapes()
    if shape_offset is None:
        shape_offset = random.randrange(4)

    # input is 4 x (5*num - 1), output is num x num
    nmax = min(6, (max_w + 1) // 5, max_h, max_w)
    nmax = max(1, nmax)
    num = _unifint(diff_lb, diff_ub, (1, nmax))

    h = 4
    w = 5 * num - 1
    remcols = [c for c in range(10) if c != bgc]

    gi = [[bgc] * w for _ in range(h)]
    ccols = []
    for k in range(num):
        col = random.choice(remcols)
        if k == 0:
            obj, outcol = mpr[shape_offset % 4]
        else:
            obj, outcol = random.choice(mpr)
        for (r, c) in obj:
            gi[r][5 * k + c] = col
        ccols.append(outcol)

    go = [[c] * num for c in ccols]
    return {
        "input": tuple(tuple(r) for r in gi),
        "output": tuple(tuple(r) for r in go),
    }


# ---------------------------------------------------------------- 3. derive
def _score_block(B):
    """The badge a 4x4 stamp earns, read off the stamp itself."""
    one   = int(len(set(B.flatten().tolist())) == 1)      # one colour     -> 2
    asym  = int(not np.array_equal(B, B[::-1, :]))        # not ud-sym     -> 4
    notch = int(int(B[0, 0]) != int(B[1, 0]))             # dented side    -> 3
    plain = int(not (one or asym or notch))               # none of those  -> 8
    return 2 * one + 4 * asym + 3 * notch + 8 * plain


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    num = (wi + 1) // 5                  # how many stamps, read from I's width
    if num < 1:
        num = 1
    labels = [_score_block(I[0:4, 5 * k:5 * k + 4]) for k in range(num)]

    ops, sels = [], []

    # The answer is a num x num board, so size the canvas to it first — every
    # later stroke is a row of that board.  Full-rectangle bbox: a canvas
    # resize acts on the whole rectangle.
    ops.append(33)
    sels.append([0, 0, num - 1, num - 1])

    # The board as it now stands: a transparent crop of I's top-left corner
    # (plus zero rows below row 3 when num > 4).
    cur = np.zeros((num, num), dtype=int)
    for r in range(min(num, hi)):
        for c in range(min(num, wi)):
            cur[r, c] = int(I[r, c])

    def lay(cells, color):
        """Paint a whole region; stay silent when it already holds that colour."""
        cells = sorted(cells)
        if not cells or all(int(cur[r, c]) == color for r, c in cells):
            return
        ops.append(int(color))
        sels.append(sel_of(cells))
        for r, c in cells:
            cur[r, c] = color

    def row(k):
        return [(k, c) for c in range(num)]

    # Which way to work through the board is the solver's own choice, so take it
    # from the strip in front of them — how many stamps there are and what
    # colours they are inked in.  ((0,0) carries ink in all four stamps.)
    # Nothing here looks at the answer: a different O yields the same route.
    stamp_cols = [int(I[0, 5 * k]) for k in range(num)]
    route = (num + sum(stamp_cols)) % 4

    if route == 0:
        # stamp by stamp, left to right down the board
        for k in range(num):
            lay(row(k), labels[k])
    elif route == 1:
        # same reading, worked backwards from the last stamp up
        for k in range(num - 1, -1, -1):
            lay(row(k), labels[k])
    elif route == 2:
        # lay the commonest badge down as a ground, then draw the odd rows over it
        base = max(set(labels), key=lambda L: (labels.count(L), -labels.index(L)))
        lay([(r, c) for r in range(num) for c in range(num)], base)
        for k in range(num):
            if labels[k] != base:
                lay(row(k), labels[k])
    else:
        # one badge at a time: all the rows that earned it, in a single stroke
        for lab in sorted(set(labels), key=labels.index):
            lay([(k, c) for k in range(num) if labels[k] == lab
                 for c in range(num)], lab)

    ops.append(34)
    sels.append([0, 0, num - 1, num - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task 995c5fa3"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 995c5fa3"
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
                                f"for task 995c5fa3"
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
                    f"Failed to build a complete episode for task 995c5fa3 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"995c5fa3-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
