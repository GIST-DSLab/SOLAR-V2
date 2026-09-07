"""
ARC Task: 4be741c5 (RE-ARC) — LLM-generated grid_maker
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


def _unifint(diff_lb, diff_ub, bounds):
    a, b = bounds
    lo = a + int((b - a) * diff_lb)
    hi = a + int((b - a) * diff_ub)
    lo = max(a, min(lo, b))
    hi = max(a, min(hi, b))
    if lo > hi:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def _dmirror(g):
    return tuple(zip(*g))


# discrete structural variant: which way the colour bands run
# (transposed=True  -> horizontal bands, column-shaped output)
# (transposed=False -> vertical bands,   row-shaped output)
VARIANTS = [{"transposed": True}, {"transposed": False}]


def sample_colors(num_examples=None) -> dict:
    palette = list(range(10))
    random.shuffle(palette)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"palette": palette, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, palette, transposed=None) -> dict:
    if transposed is None:
        transposed = random.choice([True, False])

    if transposed:
        hmax, wmax = min(30, max_w), min(30, max_h)
    else:
        hmax, wmax = min(30, max_h), min(30, max_w)
    hmax = max(4, hmax)
    wmax = max(6, wmax)

    h = _unifint(diff_lb, diff_ub, (4, hmax))
    w = _unifint(diff_lb, diff_ub, (6, wmax))
    numcolors = _unifint(diff_lb, diff_ub, (2, w // 3))

    ccols = list(palette[:numcolors])
    random.shuffle(ccols)
    go = (tuple(ccols),)

    gi = []
    for c in ccols:
        for _ in range(3):
            gi.append(tuple([c] * h))
    gi = tuple(gi)
    while len(gi) < w:
        idx = random.randint(0, len(gi) - 1)
        gi = gi[:idx] + gi[idx:idx + 1] + gi[idx:]
    gi = _dmirror(gi)                      # h x w, each column one colour band

    ndisturbances = _unifint(diff_lb, diff_ub, (0, 3 * h * numcolors))
    gi = [list(r) for r in gi]
    for _k in range(ndisturbances):
        options = []
        for a in range(h):
            for b in range(w - 3):
                if gi[a][b] == gi[a][b + 1] and gi[a][b + 2] == gi[a][b + 3]:
                    options.append((a, b, gi[a][b], gi[a][b + 2]))
        if len(options) == 0:
            break
        a, b, c1, c2 = random.choice(options)
        if random.choice((True, False)):
            gi[a][b + 1] = c2
        else:
            gi[a][b + 2] = c1
    gi = tuple(tuple(r) for r in gi)

    if transposed:
        gi = _dmirror(gi)
        go = _dmirror(go)

    return {"input": [list(r) for r in gi], "output": [list(r) for r in go]}


def derive_operations(I, O):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # --- read the rule off I ---------------------------------------------
    # The grid is a stack of solid colour bands with jagged borders.
    # If row 0 is one flat colour the bands run horizontally (read column 0),
    # otherwise they run vertically (read row 0).
    row0 = I[0, :].tolist()
    vertical_out = len(set(row0)) == 1
    line = I[:, 0].tolist() if vertical_out else row0

    # band order = order of first appearance along that line (dedupe)
    seq = []
    for v in line:
        if v not in seq:
            seq.append(v)
    k = len(seq)

    # --- stamp each band's colour at that band's rank position ------------
    # band j (an object of I) contributes exactly one cell, at index j of the
    # answer strip. Bands handled in their natural order along the line.
    for j, col in enumerate(seq):
        r, c = (j, 0) if vertical_out else (0, j)
        if int(I[r, c]) != col:          # cell already holds this band's colour
            ops.append(int(col))
            sels.append([r, c, 0, 0])

    # --- keep only the strip ---------------------------------------------
    ops.append(33)
    sels.append([0, 0, k - 1, 0] if vertical_out else [0, 0, 0, k - 1])

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
                        f"num_examples+1 ({num_examples + 1}) for task 4be741c5"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 4be741c5"
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
                                f"for task 4be741c5"
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
                    f"Failed to build a complete episode for task 4be741c5 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"4be741c5-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
