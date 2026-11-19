"""
ARC Task: b7249182 (RE-ARC) — LLM-generated grid_maker
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


VARIANTS = [{"transpose": False}, {"transpose": True}]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc, ca, cb = random.sample(cols, 3)
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "ca": ca, "cb": cb, "instance_plan": plan}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, ca=None, cb=None,
             transpose=None) -> dict:
    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        lo = a + round((b - a) * lb)
        hi = a + round((b - a) * ub)
        lo, hi = min(lo, hi), max(lo, hi)
        return random.randint(lo, hi)

    if bgc is None or ca is None or cb is None:
        bgc, ca, cb = random.sample(range(10), 3)
    if transpose is None:
        transpose = random.choice([True, False])

    # canvas dims in the "vertical" frame; transposition swaps them
    lim_h, lim_w = (max_w, max_h) if transpose else (max_h, max_w)
    h = unifint(diff_lb, diff_ub, (7, min(30, lim_h)))
    w = unifint(diff_lb, diff_ub, (5, min(30, lim_w)))
    ih = unifint(diff_lb, diff_ub, (3, max(3, (h - 1) // 2)))

    subga = [[bgc] * 5 for _ in range(ih)]
    for r in range(0, ih - 1):
        subga[r][2] = ca
    for c in range(5):
        subga[ih - 2][c] = ca
    subga[ih - 1][0] = ca
    subga[ih - 1][4] = ca
    subgb = [[cb if v == ca else v for v in row] for row in subga]
    subg = subga + subgb[::-1]

    loci = random.randint(0, h - 2 * ih)
    locj = random.randint(0, w - 5)
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    for r in range(2 * ih):
        for c in range(5):
            go[loci + r][locj + c] = subg[r][c]
    gi[loci][locj + 2] = ca
    gi[loci + 2 * ih - 1][locj + 2] = cb

    if transpose:
        gi = [list(x) for x in zip(*gi)]
        go = [list(x) for x in zip(*go)]
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # two single-cell markers on a common line
    marks = [(int(r), int(c)) for r in range(hi) for c in range(wi) if I[r, c] != bgc]
    (ra, cak), (rb, cbk) = marks[0], marks[1]

    if cak == cbk:
        vertical = True
        fixed = cak
        p_lo, p_hi = (ra, rb) if ra < rb else (rb, ra)
        col_lo = int(I[p_lo, fixed])
        col_hi = int(I[p_hi, fixed])
    else:
        vertical = False
        fixed = ra
        p_lo, p_hi = (cak, cbk) if cak < cbk else (cbk, cak)
        col_lo = int(I[fixed, p_lo])
        col_hi = int(I[fixed, p_hi])

    span = p_hi - p_lo + 1
    ih = span // 2

    def cell(p, s):
        return (p, s) if vertical else (s, p)

    # Reading: both markers shoot a line toward each other along their shared
    # axis; where the two lines meet, each line's tip opens into a bracket
    # (cross-bar + two arms) facing the other one.
    stem_lo = [cell(p, fixed) for p in range(p_lo, p_lo + ih - 2)]
    stem_hi = [cell(p, fixed) for p in range(p_lo + ih + 2, p_hi + 1)]
    bracket_lo = ([cell(p_lo + ih - 2, s) for s in range(fixed - 2, fixed + 3)]
                  + [cell(p_lo + ih - 1, fixed - 2), cell(p_lo + ih - 1, fixed + 2)])
    bracket_hi = ([cell(p_lo + ih + 1, s) for s in range(fixed - 2, fixed + 3)]
                  + [cell(p_lo + ih, fixed - 2), cell(p_lo + ih, fixed + 2)])

    G = I.copy()
    ops, sels = [], []

    def draw(cells, color):
        todo = sorted({(r, c) for (r, c) in cells if G[r, c] != color})
        if todo:
            for r, c in todo:
                G[r, c] = color
            ops.append(color)
            sels.append(sel_of(todo))

    # the two lines grow toward each other
    draw(stem_lo, col_lo)
    draw(stem_hi, col_hi)
    # at the meeting point each tip opens into its bracket
    draw(bracket_lo, col_lo)
    draw(bracket_hi, col_hi)

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
                        f"num_examples+1 ({num_examples + 1}) for task b7249182"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b7249182"
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
                                f"for task b7249182"
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
                    f"Failed to build a complete episode for task b7249182 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b7249182-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
