"""
ARC Task: b6afb2da (RE-ARC) — LLM-generated grid_maker
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
    # Background never uses 1, 2 or 4. Those are the output colours.
    cols = [c for c in range(10) if c not in (1, 2, 4)]
    bgc = random.choice(cols)
    return {"bgc": bgc}


def _unif(diff_lb, diff_ub, lo, hi):
    if hi < lo:
        hi = lo
    a = lo + int(round(diff_lb * (hi - lo)))
    b = lo + int(round(diff_ub * (hi - lo)))
    a, b = max(lo, min(a, hi)), max(lo, min(b, hi))
    if b < a:
        a, b = b, a
    return random.randint(a, b)


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, **kw) -> dict:
    cols = [c for c in range(10) if c not in (1, 2, 4)]
    if bgc is None:
        bgc = random.choice(cols)
    h = _unif(diff_lb, diff_ub, 10, max(10, min(30, max_h)))
    w = _unif(diff_lb, diff_ub, 10, max(10, min(30, max_w)))
    remcols = [c for c in cols if c != bgc]
    gi = [[bgc] * w for _ in range(h)]
    go = [[bgc] * w for _ in range(h)]
    num = _unif(diff_lb, diff_ub, 1, 9)
    free = {(i, j) for i in range(h) for j in range(w)}
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if not remcols or not free:
            break
        oh = random.randint(3, 7)
        ow = random.randint(3, 7)
        subs = [ij for ij in free if ij[0] < h - oh and ij[1] < w - ow]
        if not subs:
            tr += 1
            continue
        li, lj = random.choice(sorted(subs))
        bd = {(r, c) for r in range(li, li + oh) for c in range(lj, lj + ow)}
        col = random.choice(remcols)
        if bd.issubset(free):
            remcols.remove(col)
            for (r, c) in bd:
                gi[r][c] = col
                edge = r in (li, li + oh - 1) or c in (lj, lj + ow - 1)
                corner = r in (li, li + oh - 1) and c in (lj, lj + ow - 1)
                go[r][c] = 1 if corner else (4 if edge else 2)
            succ += 1
            free -= bd
        tr += 1
    return {"input": gi, "output": go}


def derive_operations(I, O, examples=None):
    I = np.asarray(I, dtype=int)
    hi, wi = I.shape

    # Background: the same colour in every input of the episode. Pick the colour
    # present in I that has the most votes across I and the example inputs.
    cnt = Counter(I.flatten().tolist())
    bgc = cnt.most_common(1)[0][0]
    if examples:
        votes = Counter()
        for pair in examples:
            try:
                ex = np.asarray(pair[0], dtype=int)
            except Exception:
                continue
            votes[Counter(ex.flatten().tolist()).most_common(1)[0][0]] += 1
        votes[bgc] += 1
        for col, _ in votes.most_common():
            if col in cnt:
                bgc = col
                break

    # Each cell gets a role from its own neighbourhood: how many of its four
    # orthogonal neighbours share its (non-background) colour. Every rectangle
    # has a distinct colour, so the count stays inside one rectangle.
    #   4 neighbours -> interior (2), 3 -> side (4), 2 -> corner (1).
    interior, side, corner = [], [], []
    for r in range(hi):
        for c in range(wi):
            v = I[r, c]
            if v == bgc:
                continue
            n = 0
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < hi and 0 <= cc < wi and I[rr, cc] == v:
                    n += 1
            if n == 4:
                interior.append((r, c))
            elif n == 3:
                side.append((r, c))
            elif n == 2:
                corner.append((r, c))

    ops, sels = [], []
    # One op per role class, across all rectangles at once.
    if interior:
        ops.append(2); sels.append(sel_of(interior))
    if side:
        ops.append(4); sels.append(sel_of(side))
    if corner:
        ops.append(1); sels.append(sel_of(corner))

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
                        f"num_examples+1 ({num_examples + 1}) for task b6afb2da"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task b6afb2da"
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
                                f"for task b6afb2da"
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
                    f"Failed to build a complete episode for task b6afb2da "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"b6afb2da-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
