"""
ARC Task: a9f96cdd (RE-ARC) — LLM-generated grid_maker
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

MARKS = (((-1, -1), 3), ((-1, 1), 6), ((1, -1), 8), ((1, 1), 7))


def sample_colors(num_examples=None) -> dict:
    cols = [c for c in range(10) if c not in (3, 6, 7, 8)]
    bgc = random.choice(cols)
    fgc = random.choice([c for c in cols if c != bgc])
    return {"bgc": bgc, "fgc": fgc}


def _unifint(lb, ub, rng):
    a, b = rng
    d = random.uniform(lb, ub)
    return max(a, min(b, int(round(a + (b - a) * d))))


def generate(diff_lb, diff_ub, max_h, max_w, bgc, fgc) -> dict:
    h = _unifint(diff_lb, diff_ub, (3, max(3, min(30, max_h))))
    w = _unifint(diff_lb, diff_ub, (3, max(3, min(30, max_w))))
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    locs = {(i, j) for i in range(h) for j in range(w)}
    noccs = _unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 10)))
    for _ in range(noccs):
        if not locs:
            break
        r, c = random.choice(sorted(locs))
        locs -= {(r + a, c + b) for a in range(-2, 3) for b in range(-2, 3)}
        gi[r][c] = fgc
        for (dr, dc), col in MARKS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                go[rr][cc] = col
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    """Reading (shared with the golf entry): every dot is replaced by ONE fixed 3x3
    stamp centred on it -- 3 . 6 / . bg . / 8 . 7.  The stamp is a unit: build it once
    around one dot that has room for all of it, copy it (CopyO, whole 3x3 rectangle),
    and paste it over every other dot that has room.  Dots touching an edge only get
    the part of the stamp that fits (drawn mark by mark).  Finally any dot cells still
    showing are wiped to background (needed when bg == 0, where the stamp's background
    centre is transparent to Paste).  O is used only for its shape."""
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    dots = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]
    inner = [(r, c) for r, c in dots if 0 < r < hi - 1 and 0 < c < wi - 1]
    edge = [d for d in dots if d not in inner]
    ops, sels = [], []

    def paint(col, cells):
        ops.append(int(col))
        sels.append(sel_of(cells))

    erased = set()
    if inner:
        r0, c0 = inner[0]
        # build the stamp once around the first dot that has room for all of it
        for (dr, dc), col in MARKS:
            paint(col, [(r0 + dr, c0 + dc)])
        if bgc != 0:
            paint(bgc, [(r0, c0)])
            erased.add((r0, c0))
        # copy the finished stamp: the full 3x3 rectangle around that dot
        ops.append(29); sels.append([r0 - 1, c0 - 1, 2, 2])
        # stamp it over every other dot with room (paste origin = up-left of the dot)
        for (r, c) in inner[1:]:
            ops.append(30); sels.append([r - 1, c - 1, 0, 0])
            if bgc != 0:
                erased.add((r, c))
    # dots on the border: only the marks that fit on the grid
    for (r, c) in edge:
        for (dr, dc), col in MARKS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < hi and 0 <= cc < wi:
                paint(col, [(rr, cc)])
    # remaining visible dot cells -> background
    rest = [d for d in dots if d not in erased]
    if rest:
        paint(bgc, rest)
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
                        f"num_examples+1 ({num_examples + 1}) for task a9f96cdd"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a9f96cdd"
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
                                f"for task a9f96cdd"
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
                    f"Failed to build a complete episode for task a9f96cdd "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a9f96cdd-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
