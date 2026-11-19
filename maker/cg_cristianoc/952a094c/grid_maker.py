"""
ARC Task: 952a094c (RE-ARC) — LLM-generated grid_maker
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
    # The rule only cares where the four corner markers sit, not what colour they are.
    # So fix the background and frame colours for the episode, and let each instance
    # draw its own four marker colours from what is left.
    cols = list(range(10))
    bgc, fgc = random.sample(cols, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc, **kwargs) -> dict:
    def unifint(lb, ub, rng):
        a, b = rng
        if b < a:
            b = a
        lo = a + int(round(lb * (b - a)))
        hi = a + int(round(ub * (b - a)))
        lo, hi = min(lo, hi), max(lo, hi)
        return random.randint(lo, hi)

    h = unifint(diff_lb, diff_ub, (6, max(6, max_h)))
    w = unifint(diff_lb, diff_ub, (6, max(6, max_w)))
    ih = unifint(diff_lb, diff_ub, (4, h - 2))
    iw = unifint(diff_lb, diff_ub, (4, w - 2))
    loci = random.randint(1, h - ih - 1)
    locj = random.randint(1, w - iw - 1)
    rest = [c for c in range(10) if c not in (bgc, fgc)]
    a, b, c, d = random.sample(rest, 4)

    g = np.full((h, w), bgc, dtype=int)
    r0, c0, r1, c1 = loci, locj, loci + ih - 1, locj + iw - 1
    g[r0, c0:c1 + 1] = fgc
    g[r1, c0:c1 + 1] = fgc
    g[r0:r1 + 1, c0] = fgc
    g[r0:r1 + 1, c1] = fgc
    gi = g.copy()
    go = g.copy()
    gi[r0 + 1, c0 + 1] = a; go[r1 + 1, c1 + 1] = a
    gi[r0 + 1, c1 - 1] = b; go[r1 + 1, c0 - 1] = b
    gi[r1 - 1, c0 + 1] = c; go[r0 - 1, c1 + 1] = c
    gi[r1 - 1, c1 - 1] = d; go[r0 - 1, c0 - 1] = d
    return {"input": gi.tolist(), "output": go.tolist()}


def derive_operations(I, O, examples=None):
    # How I read it (same as the golf entry): each of the four inner corner markers
    # swaps places with the empty cell diagonally outside the opposite corner of the
    # frame. That makes each marker one object that moves. It is the same cell with the
    # same colour, just somewhere else. So each marker gets one grab, a chain of unit
    # Moves, and then its original cell is set back to bgc. That bgc is what the swap
    # puts there.
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape
    ops, sels = [], []

    ranked = Counter(I.flatten().tolist()).most_common()
    bgc = ranked[0][0]
    fgc = ranked[1][0]          # the frame has at least 12 cells; each marker has 1

    rows, cols = np.where(I == fgc)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())

    pairs = [
        ((r0 + 1, c0 + 1), (r1 + 1, c1 + 1)),  # inner TL -> outside BR
        ((r0 + 1, c1 - 1), (r1 + 1, c0 - 1)),  # inner TR -> outside BL
        ((r1 - 1, c0 + 1), (r0 - 1, c1 + 1)),  # inner BL -> outside TR
        ((r1 - 1, c1 - 1), (r0 - 1, c0 - 1)),  # inner BR -> outside TL
    ]

    for (sr, sc), (tr, tc) in pairs:
        col = int(I[sr, sc])
        if col == bgc:
            continue
        if col == 0:
            # ARCLE Move cannot carry a 0 cell, so this marker is painted instead
            ops.append(int(bgc)); sels.append(sel_of([(sr, sc)]))
            ops.append(0); sels.append(sel_of([(tr, tc)]))
            continue
        dr, dc = tr - sr, tc - sc
        steps = [(20 if dr < 0 else 21)] * abs(dr) + [(23 if dc < 0 else 22)] * abs(dc)
        first = True
        for op in steps:
            ops.append(op)
            sels.append(sel_of([(sr, sc)]) if first else sel_of([]))
            first = False
        # the only cell left at 0 is where the marker started; set it to bgc, as in a swap
        if bgc != 0:
            ops.append(int(bgc)); sels.append(sel_of([(sr, sc)]))

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
                        f"num_examples+1 ({num_examples + 1}) for task 952a094c"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 952a094c"
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
                                f"for task 952a094c"
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
                    f"Failed to build a complete episode for task 952a094c "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"952a094c-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
