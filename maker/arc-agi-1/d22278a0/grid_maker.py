"""
ARC Task: d22278a0 (RE-ARC) — LLM-generated grid_maker
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
def sample_colors(num_examples=None) -> dict:
    cols = list(range(10))
    bgc = random.choice(cols)
    variants = [{"ncorns": 1}, {"ncorns": 2}, {"ncorns": 3}, {"ncorns": 4}]
    n_ex = num_examples if num_examples else 3
    if n_ex >= len(variants):
        examples = [dict(v) for v in variants]
        examples += [dict(random.choice(variants)) for _ in range(n_ex - len(variants))]
        random.shuffle(examples)
    else:
        examples = [dict(v) for v in random.sample(variants, n_ex)]
    plan = examples + [dict(random.choice(examples))]
    return {"bgc": bgc, "instance_plan": plan}


def generate(diff_lb: float, diff_ub: float, max_h: int, max_w: int, bgc: int,
             ncorns: int = None) -> dict:
    cols = interval(0, 10, 1)
    if ncorns is None:
        ncorns = random.choice([1, 2, 3, 4])
    h = unifint(diff_lb, diff_ub, (4, max_h))
    w = unifint(diff_lb, diff_ub, (4, max_w))
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    crns = corners(inds)
    crns = sample(totuple(crns), ncorns)
    ccols = sample(remcols, ncorns)
    for col, crn in zip(ccols, crns):
        gi = fill(gi, col, {crn})
        go = fill(go, col, {crn})
        rings = {crn}
        for k in range(1, max(h, w) // 2 + 2, 1):
            rings = rings | outbox(outbox(rings))
        if len(crns) > 1:
            ff = lambda ij: manhattan({ij}, {crn}) < min(
                apply(rbind(manhattan, {ij}), apply(initset, remove(crn, crns))))
        else:
            ff = lambda ij: True
        locs = sfilter(inds, ff) & rings
        go = fill(go, col, locs)
    return {'input': gi, 'output': go}


def derive_operations(I, O):
    import numpy as np
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape

    # Input is a plain canvas with only corner cells colored -> an interior cell is bgc.
    bgc = int(I[1, 1])

    # Seeds: the corner cells that carry a non-background color.
    seeds = []
    for (r, c) in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        if int(I[r, c]) != bgc:
            seeds.append((r, c, int(I[r, c])))

    ops, sels = [], []
    maxrad = max(h - 1, w - 1)

    for (r0, c0, col) in seeds:
        others = [(r, c) for (r, c, _) in seeds if (r, c) != (r0, c0)]
        dr = 1 if r0 == 0 else -1
        dc = 1 if c0 == 0 else -1

        def owns(r, c):
            # cell belongs to this seed only if strictly nearest (L1); ties stay background
            if not (0 <= r < h and 0 <= c < w):
                return False
            d = abs(r - r0) + abs(c - c0)
            for (r1, c1) in others:
                if d >= abs(r - r1) + abs(c - c1):
                    return False
            return True

        def emit_runs(cells, vertical):
            run = []
            for cell in cells + [None]:
                if cell is not None and owns(cell[0], cell[1]):
                    run.append(cell)
                    continue
                if run:
                    rs = [p[0] for p in run]
                    cs = [p[1] for p in run]
                    if vertical:
                        sels.append([min(rs), cs[0], max(rs) - min(rs), 0])
                    else:
                        sels.append([rs[0], min(cs), 0, max(cs) - min(cs)])
                    ops.append(col)
                    run = []

        # concentric square rings around this seed, radius 2, 4, 6, ... outward.
        # Only the two arms facing into the grid exist (seed sits on a corner).
        k = 2
        while k <= maxrad:
            cc = c0 + k * dc
            emit_runs([(r0 + i * dr, cc) for i in range(k + 1)], True)
            rr = r0 + k * dr
            emit_runs([(rr, c0 + j * dc) for j in range(k)], False)
            k += 2

    ops.append(34)
    sels.append([0, 0, h - 1, w - 1])
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
                        f"num_examples+1 ({num_examples + 1}) for task d22278a0"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task d22278a0"
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
                                f"for task d22278a0"
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
                    f"Failed to build a complete episode for task d22278a0 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"d22278a0-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
