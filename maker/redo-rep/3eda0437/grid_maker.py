"""
ARC Task: 3eda0437 (RE-ARC) — LLM-generated grid_maker
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


def sample_colors(num_examples=None) -> dict:
    # 0 = the "hole" color (hardcoded), 6 = the marker color (hardcoded) -> neither is sampled.
    # Only the random foreground palette is episode-level; the rule itself is color-independent.
    cols = [c for c in range(1, 10) if c != 6]
    ncols = random.randint(2, 8)
    fgcs = random.sample(cols, ncols)
    return {"fgcs": fgcs}


def generate(diff_lb, diff_ub, max_h, max_w, fgcs=None) -> dict:
    cols = [c for c in range(1, 10) if c != 6]
    if not fgcs:
        fgcs = random.sample(cols, random.randint(2, 8))
    fgcs = [c for c in fgcs if c in cols] or [1]

    def unifint(lb, ub, bounds):
        a, b = bounds
        if b < a:
            b = a
        ilb = a + int((b - a) * lb)
        iub = a + int((b - a) * ub)
        ilb = max(a, min(ilb, b))
        iub = max(ilb, min(iub, b))
        return random.randint(ilb, iub)

    def max_zero_rect(g, h, w):
        """all maximal-area all-zero rectangles (a,b >= 2) and their occurrences"""
        z = (g == 0)
        ps = np.zeros((h + 1, w + 1), dtype=int)
        ps[1:, 1:] = np.cumsum(np.cumsum(z.astype(int), axis=0), axis=1)
        best, mapper = 0, {}
        for a in range(2, h + 1):
            for b in range(2, w + 1):
                if a * b < best:
                    continue
                tot = (ps[a:, b:] - ps[:h - a + 1, b:]
                       - ps[a:, :w - b + 1] + ps[:h - a + 1, :w - b + 1])
                rs, cs = np.nonzero(tot == a * b)
                if len(rs) == 0:
                    continue
                occs = sorted((int(r), int(c)) for r, c in zip(rs, cs))
                if a * b > best:
                    best, mapper = a * b, {(a, b): occs}
                else:
                    mapper[(a, b)] = occs
        return mapper

    hcap, wcap = max(3, int(max_h)), max(3, int(max_w))
    gi = None
    for attempt in range(400):
        want_multi = attempt < 300
        if random.choice((True, False)):
            h = unifint(diff_lb, diff_ub, (3, min(8, hcap)))
            w = unifint(diff_lb, diff_ub, (3, min(30, wcap)))
        else:
            h = unifint(diff_lb, diff_ub, (3, min(30, hcap)))
            w = unifint(diff_lb, diff_ub, (3, min(8, wcap)))
        gi = np.array([[random.choice(fgcs) for _ in range(w)] for _ in range(h)], dtype=int)
        spac = unifint(diff_lb, diff_ub, (1, (h * w) // 3 * 2))
        inds = [(r, c) for r in range(h) for c in range(w)]
        for (r, c) in random.sample(inds, spac):
            gi[r, c] = 0
        lr, lc = random.randint(0, h - 1), random.randint(0, w - 1)
        for (r, c) in ((lr, lc), (lr, lc + 1), (lr + 1, lc), (lr + 1, lc + 1)):
            if 0 <= r < h and 0 <= c < w:
                gi[r, c] = 0
        mapper = max_zero_rect(gi, h, w)
        if len(mapper) != 1:                 # a size tie between two shapes is ambiguous
            continue
        (a, b), occs = next(iter(mapper.items()))
        if want_multi and len(occs) < 2:     # prefer instances that actually replicate
            continue
        go = gi.copy()
        for (r, c) in occs:
            go[r:r + a, c:c + b] = 6
        return {"input": gi.tolist(), "output": go.tolist()}
    return {"input": gi.tolist(), "output": gi.tolist()}


def derive_operations(I, O):
    from maker.sel_helpers import sel_of
    I = np.asarray(I, dtype=int)
    O = np.asarray(O, dtype=int)
    h, w = I.shape
    ops, sels = [], []

    # --- the rule, measured from I alone: the largest all-zero rectangle (sides >= 2),
    #     then every place that same rectangle occurs. ---
    z = (I == 0)
    ps = np.zeros((h + 1, w + 1), dtype=int)
    ps[1:, 1:] = np.cumsum(np.cumsum(z.astype(int), axis=0), axis=1)
    best, shapes = 0, {}
    for a in range(2, h + 1):
        for b in range(2, w + 1):
            if a * b < best:
                continue
            tot = (ps[a:, b:] - ps[:h - a + 1, b:]
                   - ps[a:, :w - b + 1] + ps[:h - a + 1, :w - b + 1])
            rs, cs = np.nonzero(tot == a * b)
            if len(rs) == 0:
                continue
            occ = sorted((int(r), int(c)) for r, c in zip(rs, cs))
            if a * b > best:
                best, shapes = a * b, {(a, b): occ}
            else:
                shapes[(a, b)] = occ

    def cells_of(r, c, a, b):
        return [(rr, cc) for rr in range(r, r + a) for cc in range(c, c + b)]

    target = {(int(r), int(c)) for r in range(h) for c in range(w)
              if O[r, c] == 6 and I[r, c] != 6}

    occs, predicted, a, b = [], set(), 0, 0
    if len(shapes) == 1:
        (a, b), occs = next(iter(shapes.items()))
        for (r, c) in occs:
            predicted.update(cells_of(r, c, a, b))

    if occs and predicted == target:
        cur = I.copy()
        # 1) mark the largest all-zero rectangle itself (first occurrence, raster order)
        r0, c0 = occs[0]
        ops.append(6)
        sels.append(sel_of(cells_of(r0, c0, a, b)))     # exactly the a*b cells of the block
        cur[r0:r0 + a, c0:c0 + b] = 6
        if len(occs) > 1:
            # 2) that freshly painted block is the stamp -> clipboard (working grid, not input:
            #    in the input those cells are 0 and 0 copies as "nothing")
            ops.append(29)
            sels.append(sel_of(cells_of(r0, c0, a, b)))
            # 3) replicate it at every other occurrence of the same rectangle
            for (r, c) in occs[1:]:
                blk = cells_of(r, c, a, b)
                if all(cur[rr, cc] == 6 for rr, cc in blk):
                    continue                            # already covered: paste would do nothing
                ops.append(30)
                sels.append(sel_of([(r, c)]))           # paste origin = block's top-left
                cur[r:r + a, c:c + b] = 6
    elif target:
        # safety net (unique-shape rule not reproducible): mark the cells the rule marks
        ops.append(6)
        sels.append(sel_of(sorted(target)))

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
                        f"num_examples+1 ({num_examples + 1}) for task 3eda0437"
                    )
                if instance_plan is not None:
                    if len(instance_plan) != num_examples + 1:
                        raise ValueError(
                            f"instance_plan length {len(instance_plan)} != "
                            f"num_examples+1 ({num_examples + 1}) for task 3eda0437"
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
                                f"for task 3eda0437"
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
                    f"Failed to build a complete episode for task 3eda0437 "
                    f"after 5 attempts"
                )

            dataset.append((ex_in, ex_out, pr_in, pr_out, {
                "id":         f"3eda0437-rearc-llm_{_sn + 1}",
                "concept":    "RE-ARC LLM",
                "operations": ops,
                "selections": sels,
            }))

        return dataset
