"""
ARC Task: a5f85a15 (RE-ARC) — LLM-generated grid_maker
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
import numpy as np
from collections import Counter
from maker.sel_helpers import sel_of


def sample_colors(num_examples=None) -> dict:
    # The generator samples exactly two colours at random (bgc, fgc); the mark
    # colour 4 is hardcoded, so it is NOT sampled here.
    colopts = [c for c in range(10) if c != 4]
    bgc, fgc = random.sample(colopts, 2)
    return {"bgc": bgc, "fgc": fgc}


def generate(diff_lb, diff_ub, max_h, max_w, bgc=None, fgc=None) -> dict:
    h = unifint(diff_lb, diff_ub, (3, max_h))
    w = unifint(diff_lb, diff_ub, (3, max_w))
    startlocs = apply(toivec, interval(h - 1, 0, -1)) + apply(tojvec, interval(0, w, 1))
    cands = interval(0, h + w - 1, 1)
    num = unifint(diff_lb, diff_ub, (1, (h + w - 1) // 3))
    locs = []
    for k in range(num):
        if len(cands) == 0:
            break
        loc = choice(cands)
        locs.append(loc)
        cands = remove(loc, cands)
        cands = remove(loc - 1, cands)
        cands = remove(loc + 1, cands)
    locs = set([startlocs[loc] for loc in locs])
    if bgc is None or fgc is None:
        colopts = remove(4, interval(0, 10, 1))
        bgc, fgc = sample(colopts, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    for loc in locs:
        ln = order(shoot(loc, (1, 1)), first)
        gi = fill(gi, fgc, ln)
        go = fill(go, fgc, ln)
        go = fill(go, 4, ln[1::2])
    return {'input': gi, 'output': go}


def derive_operations(I, O, examples=None):
    """
    Rule: the grid holds parallel down-right diagonal rays, each starting on the
    top row or the left column.  Along every ray a two-cell motif repeats with
    period 2: <line colour> then <mark colour>.  So the account is: build the
    motif once (one Color op on the very first mark cell), copy that 2x2 motif,
    and stamp it along every ray, cell-pair by cell-pair, ray by ray.
    """
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    hi, wi = I.shape
    ho, wo = O.shape
    ops, sels = [], []

    # --- mark colour: the colour the demonstrations introduce (never in their inputs)
    mark = None
    if examples:
        for ex in examples:
            try:
                ex_i, ex_o = ex[0], ex[1]
            except Exception:
                ex_i, ex_o = ex["input"], ex["output"]
            a = set(np.asarray(ex_i).flatten().tolist())
            b = set(np.asarray(ex_o).flatten().tolist())
            new = b - a
            if len(new) == 1:
                mark = int(next(iter(new)))
                break
    if mark is None:
        mark = 4

    # --- background = the colour the canvas is painted with (the dominant one;
    #     rays occupy at most a third of the cells by construction)
    bgc = Counter(I.flatten().tolist()).most_common(1)[0][0]

    # --- recover the rays: every line lives on its own diagonal (c - r), and no
    #     two lines share adjacent diagonals, so grouping by offset is exact.
    by_diag = {}
    for r in range(hi):
        for c in range(wi):
            if I[r, c] != bgc:
                by_diag.setdefault(c - r, []).append((r, c))
    rays = [sorted(cells) for cells in by_diag.values()]
    rays.sort(key=lambda cells: cells[0])          # ray order = start in reading order

    # --- the marked cells of each ray: odd positions counted from the ray's start
    ray_targets = [(ray, ray[1::2]) for ray in rays]
    total = sum(len(t) for _, t in ray_targets)

    if total == 0:
        # every ray is a single cell: nothing repeats, nothing to mark
        ops.append(34); sels.append([0, 0, ho - 1, wo - 1])   # whole grid
        return ops, sels

    # --- 1. create the motif: paint the first ray's first mark cell
    seed_ray_idx, seed_cell = None, None
    for idx, (ray, tgts) in enumerate(ray_targets):
        if tgts:
            seed_ray_idx, seed_cell = idx, tgts[0]
            break
    sr, sc = seed_cell[0] - 1, seed_cell[1] - 1          # the ray cell just before it
    ops.append(mark); sels.append(sel_of([seed_cell]))

    # --- 2. pick that motif up: the 2x2 block <line colour> / <mark colour>
    if total > 1:
        rect = [(sr, sc), (sr, sc + 1), (sr + 1, sc), (sr + 1, sc + 1)]  # exactly this full 2x2 rectangle
        ops.append(29); sels.append(sel_of(rect))

        # --- 3. stamp it along each ray, walking outward from the ray's start
        for idx, (ray, tgts) in enumerate(ray_targets):
            for t in tgts:
                if idx == seed_ray_idx and t == seed_cell:
                    continue                                   # this one is the original
                ops.append(30); sels.append(sel_of([(t[0] - 1, t[1] - 1)]))

    ops.append(34); sels.append([0, 0, ho - 1, wo - 1])   # whole grid
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
                        f"num_examples+1 ({num_examples + 1}) for task a5f85a15"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task a5f85a15"
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
                                f"for task a5f85a15"
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
                    f"Failed to build a complete episode for task a5f85a15 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"a5f85a15-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
