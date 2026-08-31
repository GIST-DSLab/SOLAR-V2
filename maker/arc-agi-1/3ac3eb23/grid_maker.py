"""
ARC Task: 3ac3eb23 (RE-ARC) — LLM-generated grid_maker
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

VARIANTS = [
    {"mf_name": "identity"},
    {"mf_name": "rot90"},
    {"mf_name": "rot180"},
    {"mf_name": "rot270"},
]


def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    # foreground colors kept non-zero: 0 is "transparent" to ARCLE Copy/Paste,
    # and the fan is duplicated with CopyO/Paste.
    fgcols = [c for c in range(1, 10) if c != bgc]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(VARIANTS):
        examples = [dict(v) for v in VARIANTS]
        examples += [dict(random.choice(VARIANTS)) for _ in range(n_ex - len(VARIANTS))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(VARIANTS, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "fgcols": fgcols, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int,
             bgc=None, fgcols=None, mf_name=None) -> dict:
    if mf_name is None:
        mf_name = choice(("identity", "rot90", "rot180", "rot270"))
    mf = {"identity": identity, "rot90": rot90, "rot180": rot180, "rot270": rot270}[mf_name]
    # rot90/rot270 transpose the canvas -> sample within the swapped bounds
    if mf_name in ("rot90", "rot270"):
        hub, wub = max_w, max_h
    else:
        hub, wub = max_h, max_w
    h = unifint(diff_lb, diff_ub, (3, max(3, hub)))
    w = unifint(diff_lb, diff_ub, (3, max(3, wub)))
    if bgc is None:
        bgc = choice(interval(0, 10, 1))
    if fgcols is None:
        fgcols = [c for c in range(1, 10) if c != bgc]
    nlocs = unifint(diff_lb, diff_ub, (1, max(1, (w - 2) // 3)))
    locopts = interval(1, w - 1, 1)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    for k in range(nlocs):
        if len(locopts) == 0:
            break
        locj = choice(locopts)
        locopts = difference(locopts, interval(locj - 2, locj + 3, 1))
        col = choice(tuple(fgcols))
        gi = fill(gi, col, {(0, locj)})
        go = fill(go, col, {(p, locj) for p in interval(0, h, 2)})
        go = fill(go, col, {(p, locj - 1) for p in interval(1, h, 2)})
        go = fill(go, col, {(p, locj + 1) for p in interval(1, h, 2)})
    gi = mf(gi)
    go = mf(go)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    """
    Rule (read off I): every non-bg cell of I is a seed sitting on ONE border of the
    grid. Each seed grows a fan straight into the grid: at even depth d the seed's own
    lane is coloured, at odd depth the two neighbouring lanes are coloured. So the fan
    is the seed's colour repeated with PERIOD 2 in depth. Per seed we therefore
    (1) paint the depth-1 wings, giving one 2-deep unit, and (2) copy that unit and
    stamp it down the strip. O is never consulted.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape

    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]
    seeds = [(r, c) for r in range(hi) for c in range(wi) if I[r, c] != bgc]

    # which border carries the seeds -> u = inward direction, v = along-border direction
    if all(r == 0 for r, _ in seeds):
        u, v, D = (1, 0), (0, 1), hi
    elif all(r == hi - 1 for r, _ in seeds):
        u, v, D = (-1, 0), (0, 1), hi
    elif all(c == 0 for _, c in seeds):
        u, v, D = (0, 1), (1, 0), wi
    else:
        u, v, D = (0, -1), (1, 0), wi

    def cell(s, d, t):
        return (s[0] + u[0] * d + v[0] * t, s[1] + u[1] * d + v[1] * t)

    ops, sels = [], []
    for s in seeds:
        col = int(I[s[0], s[1]])

        # step 1: grow one step out of the seed -> the two wings at depth 1
        for t in (-1, 1):
            r, c = cell(s, 1, t)
            ops.append(col)
            sels.append([r, c, 0, 0])

        P = D // 2                       # number of complete 2-deep units in the strip
        if col != 0 and P > 1:
            # step 2: the depth 0..1 unit is now complete -> repeat it down the strip
            unit = [cell(s, 0, 0), cell(s, 1, -1), cell(s, 1, 1)]
            r0 = min(p[0] for p in unit)
            c0 = min(p[1] for p in unit)
            r1 = max(p[0] for p in unit)
            c1 = max(p[1] for p in unit)
            ops.append(29)
            sels.append([r0, c0, r1 - r0, c1 - c0])
            for k in range(1, P):
                ops.append(30)
                sels.append([r0 + u[0] * 2 * k, c0 + u[1] * 2 * k, 0, 0])
            if D % 2 == 1:               # one lone even depth left at the far border
                r, c = cell(s, D - 1, 0)
                ops.append(col)
                sels.append([r, c, 0, 0])
        else:
            # colour 0 is invisible to Copy/Paste (and tiny strips need no unit):
            # grow the fan outward depth by depth instead
            for d in range(2, D):
                for t in ((0,) if d % 2 == 0 else (-1, 1)):
                    r, c = cell(s, d, t)
                    ops.append(col)
                    sels.append([r, c, 0, 0])

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
                        f"num_examples+1 ({num_examples + 1}) for task 3ac3eb23"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3ac3eb23"
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
                                f"for task 3ac3eb23"
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
                    f"Failed to build a complete episode for task 3ac3eb23 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3ac3eb23-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
